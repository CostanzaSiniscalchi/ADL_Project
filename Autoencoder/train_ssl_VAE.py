import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
from data_loader_ssl import MRIDataLoader, split_data
from monai.networks.nets import VarAutoEncoder
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
import cv2


def make_viz(epoch, save_dir, count, original, scans, preds):
    os.makedirs(save_dir, exist_ok=True)

    batch_ind = 0
    mid_slice = original.shape[2] // 2  # middle depth slice (axis=2 is depth)

    rows = []
    for s in range(3):  # For each of the 3 scans
        orig_slice = original[batch_ind, s, mid_slice].cpu().numpy()
        scan_slice = scans[batch_ind, s, mid_slice].cpu().numpy()
        pred_slice = preds[batch_ind, s, mid_slice].cpu().detach().numpy()

        # Normalize each slice to [0, 255] for display
        def normalize(img):
            # img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            return (img * 255).astype(np.uint8)

        row = np.concatenate([
            normalize(scan_slice),
            normalize(orig_slice),
            normalize(pred_slice)
        ], axis=1)  # horizontally stack [input | target | prediction]
        rows.append(row)

    # vertically stack 3 scan slices
    final_image = np.concatenate(rows, axis=0)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{count}.png")
    cv2.imwrite(save_path, final_image)


def train_ssl(model, dataloader, val_dataloader, optimizer, criterion, epochs=50, patience=5, scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    target_beta = 1.0
    for epoch in range(epochs):
        beta = min(1.0, epoch / 10.0) * target_beta
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            scans = batch["numpy"].to(device)        # (B, 3, 32, 256, 240)
            original = batch["original"].to(device)  # (B, 3, 32, 256, 240)

            recon_batch, mu, log_var, _ = model(scans)

            loss = criterion(recon_batch, scans, mu, log_var, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize on first step:
            if total_loss == 0:
                make_viz(epoch, './training_viz/', 0,
                         original, scans, recon_batch)
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                scans = batch["numpy"].to(device)
                original = batch["original"].to(device)

                recon, mu, log_var, _ = model(scans)
                # sum up batch loss
                loss = criterion(recon, original, mu, log_var, beta)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(
            f"[SSL] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # allow 10 epoch warmup
        if avg_val_loss < best_val_loss and epoch > 10:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_VAE.pt')
            epochs_no_improve = 0
        elif epoch > 10:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandGaussianNoise(prob=0.2, std=0.01),
        RandAffine(prob=0.3, rotate_range=(0.05, 0.05, 0.05)),
        ToTensor()
    ])
    val_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor()
    ])

    data_root = '../data/stripped_3_scans/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

    train_set = MRIDataLoader(data_root, train_ids,
                              transform=ssl_transforms, mask_scan=True)
    val_set = MRIDataLoader(
        data_root, val_ids, transform=val_transforms, mask_scan=True)

    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = VarAutoEncoder(
        spatial_dims=3,
        in_shape=[3, 32, 256, 240],
        out_channels=3,
        latent_size=128,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        use_sigmoid=True
    )

    # recon_loss = nn.MSELoss(reduction='mean')

    # def loss_function(recon_x, x, mu, log_var, beta):
    #     mse = recon_loss(recon_x, x)
    #     kld = -0.5 * beta * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    #     return mse + kld

    ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

    def loss_function(recon_x, x, mu, log_var, beta):
        ssim_loss = ssim(recon_x, x)
        kld = -0.5 * beta * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return ssim_loss + kld
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_ssl(
        model, train_loader, val_loader, optimizer, criterion, epochs=500, patience=20)
