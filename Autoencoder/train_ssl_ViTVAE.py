import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
from data_loader_ssl import MRIDataLoader, split_data
from model import ViTAutoEnc
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
import cv2
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            img = np.clip(img, 0.0, 1.0)
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
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            scans = batch["numpy"].to(device)        # (B, 3, 32, 256, 240)
            original = batch["original"].to(device)  # (B, 3, 32, 256, 240)

            recon_batch, hidden_states = model(scans)
            # recon_batch = torch.sigmoid(recon_batch)

            loss = criterion(recon_batch, scans)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize on first step:
            if total_loss == 0:
                make_viz(epoch, './training_viz_vit_do_nosig_random_2/', 0,
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

                recon, hidden_states = model(scans)
                # recon = torch.sigmoid(recon)
                # sum up batch loss
                loss = criterion(recon, original)
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
            }, 'best_ViTVAE_2.pt')
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
    sys.stdout = open('train_ssl_ViTVAE_random_2.log', 'w')
    sys.stderr = sys.stdout
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
                              transform=ssl_transforms, mask_scan='random', mask_ratio=0.5)
    val_set = MRIDataLoader(
        data_root, val_ids, transform=val_transforms, mask_scan='random', mask_ratio=0.5)

    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = ViTAutoEnc(in_channels=3, out_channels=3, patch_size=(16, 16, 16), spatial_dims=3,
                       img_size=(32, 256, 240), proj_type='conv', dropout_rate=0.2)
    model.to(device)

    recon_loss = nn.MSELoss(reduction='mean')
    ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

    def loss_function(recon_x, x,):
        mse = recon_loss(recon_x, x)
        ssim_loss = ssim(recon_x, x)
        return 0.5*mse + 0.5*ssim_loss

    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # if os.path.exists('./best_ViTVAE.pt'):
    #     # load the state
    #     state = torch.load('./best_ViTVAE.pt',
    #                        weights_only=True, map_location=device)
    #     model.load_state_dict(state['model_state_dict'])
    #     optimizer.load_state_dict(state['optimizer_state_dict'])

    trained_model = train_ssl(
        model, train_loader, val_loader, optimizer, criterion, epochs=500, patience=10)
