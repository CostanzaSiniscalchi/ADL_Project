import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
from data_loader_ssl import MRISliceGeneratorDataLoader, split_data
from model import ViTAutoEnc
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
import cv2
from monai.transforms import Compose, NormalizeIntensity, Resize, ToTensor, RandFlip, RandAffine, RandGaussianNoise
import sys
import math
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers import Conv




def make_viz(epoch, save_dir, count, targets, inputs, preds):
    save_viz_dir = os.path.join(save_dir, 'viz_val/')
    os.makedirs(save_viz_dir, exist_ok=True)

    batch_ind = 0

    first_slice = inputs[batch_ind, 0].cpu().numpy()
    second_slice = inputs[batch_ind, 1].cpu().numpy()
    third_slice = inputs[batch_ind, 2].cpu().numpy()
    fourth_slice = inputs[batch_ind, 3].cpu().numpy()
    fifth_slice = targets[batch_ind, 0].cpu().numpy()
    fifth_pred_slice = preds[batch_ind, 0].cpu().detach().numpy()

    # Normalize each slice to [0, 255] for display
    def normalize(img):
        # img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    row = np.concatenate([
        normalize(first_slice),
        normalize(second_slice),
        normalize(third_slice),
        normalize(fourth_slice),
        normalize(fifth_slice),
        normalize(fifth_pred_slice),
    ], axis=1)  # horizontally stack [input | target | prediction]

    save_path = os.path.join(save_viz_dir, f"epoch_{epoch}_sample_{count}.png")
    cv2.imwrite(save_path, row)


def train_pred(model, dataloader, val_dataloader, optimizer, criterion, epochs=50, patience=5, scheduler=None):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch["input"].to(device)        # (B, 4, 32, 256, 240)
            targets = batch["target"].to(device)  # (B, 1, 32, 256, 240)

            recon_batch, hidden_states = model(inputs)

            recon_batch, hidden_states = model(inputs)
            recon_batch = model.relu(model.decoder_conv1(recon_batch))
            recon_batch = model.relu(model.decoder_conv2(recon_batch))
            recon_batch = model.decoder_conv3(recon_batch)
            # recon_batch = torch.sigmoid(recon_batch)

            loss = criterion(recon_batch, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # (B, 4, 32, 256, 240)
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)  # (B, 1, 32, 256, 240)

                recon_batch, hidden_states = model(inputs)
                recon_batch = model.relu(model.decoder_conv1(recon_batch))
                recon_batch = model.relu(model.decoder_conv2(recon_batch))
                recon_batch = model.decoder_conv3(recon_batch)

                
                # recon = torch.sigmoid(recon)
                # sum up batch loss
                loss = criterion(recon_batch, targets)
                # visualize on first step:
                if val_loss == 0:
                    make_viz(epoch, f'./training_runs_conv/pred_vitvae_{hidden_size_train}_{mlp_size_train}', 0,
                             targets, inputs, recon_batch)
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
            }, f'./training_runs_conv/pred_vitvae_{hidden_size_train}_{mlp_size_train}/best_{hidden_size_train}_{mlp_size_train}.pt')
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
    hidden_size_train = int(768)
    mlp_size_train = int(3072)
    
    exp_dir = f'./training_runs_conv/pred_vitvae_{hidden_size_train}_{mlp_size_train}/'
    os.makedirs(exp_dir, exist_ok=True)
    # sys.stdout = open(
    #     f'./training_runs/pred_vitvae_{hidden_size_train}_{mlp_size_train}/log_{hidden_size_train}_{mlp_size_train}.log', 'w')
    # sys.stderr = sys.stdout
    device = torch.device("cuda:0")

    

    ssl_transforms = Compose([
        lambda x: x[np.newaxis, ...],  # (1, H, W)
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandGaussianNoise(prob=0.2, std=0.01),
        RandAffine(prob=0.3, rotate_range=(0.05, 0.05, 0.05)),
        Resize((224, 224)),
        ToTensor(),
        lambda x: x.squeeze(0)
    ])
    val_transforms = Compose([
        lambda x: x[np.newaxis, ...],
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize((224, 224)),
        ToTensor(),
        lambda x: x.squeeze(0)
    ])

    print("yo")

    


    data_root = '../stripped_5_scans_slices/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

    train_set = MRISliceGeneratorDataLoader(data_root, train_ids,
                                            transform=ssl_transforms, mask_scan='random', mask_ratio='random')
    val_set = MRISliceGeneratorDataLoader(
        data_root, val_ids, transform=val_transforms, mask_scan='random', mask_ratio='random')

    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = ViTAutoEnc(in_channels=3, out_channels=3, patch_size=(16, 16), spatial_dims=2,
                       img_size=(224, 224), proj_type='conv', dropout_rate=0.2, hidden_size=hidden_size_train, mlp_dim=mlp_size_train)

    # load the state
    # state = torch.load(f'./training_runs/vitvae_{hidden_size_train}_{mlp_size_train}/best_{hidden_size_train}_{mlp_size_train}.pt',
    #                    weights_only=True, map_location=device)

    state = torch.load(f'best_{hidden_size_train}_{mlp_size_train}.pt', weights_only=True, map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=False)

    # now change to be 4 channel input and 1 channel output
    model.patch_embedding = PatchEmbeddingBlock(
        in_channels=4,
        img_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        proj_type='conv',
        dropout_rate=0.2,
        spatial_dims=2,
        hidden_size=hidden_size_train,
    )
    conv_trans = Conv[Conv.CONVTRANS, model.spatial_dims]
    up_kernel_size = [int(math.sqrt(i)) for i in model.patch_size]
    model.conv3d_transpose_1 = conv_trans(
        in_channels=16, out_channels=1, kernel_size=up_kernel_size, stride=up_kernel_size
    )

    model.decoder_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    model.decoder_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
    model.decoder_conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    model.relu = nn.ReLU()

    model.to(device)

    recon_loss = nn.MSELoss(reduction='mean')
    ssim = SSIMLoss(spatial_dims=2, data_range=1.0)

    def loss_function(recon_x, x,):
        mse = recon_loss(recon_x, x)
        ssim_loss = ssim(recon_x, x)
        return 0.5*mse + 0.5*ssim_loss

 

    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_pred(
        model, train_loader, val_loader, optimizer, criterion, epochs=500, patience=15)
