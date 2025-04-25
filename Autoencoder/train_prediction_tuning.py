import os
import time
import csv
import math
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers import Conv
from data_loader_ssl import MRIGenerationLoader, split_data
from model import ViTAutoEnc

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = '../data/stripped_5_scans/'
train_ids, test_ids, val_ids = split_data(os.listdir(data_root))
val_transforms = Compose([ScaleIntensity(minv=0.0, maxv=1.0), ToTensor()])
recon_loss = nn.MSELoss(reduction='mean')
ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

# === Loss Function ===
def loss_fn(recon_x, x):
    return 0.5 * recon_loss(recon_x, x) + 0.5 * ssim(recon_x, x)

# === Objective Function for Optuna ===
def objective(trial):
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 12])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    train_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandGaussianNoise(prob=0.2, std=0.01),
        RandAffine(prob=0.3, rotate_range=(0.05, 0.05, 0.05)),
        ToTensor()
    ])
    train_set = MRIGenerationLoader(data_root, train_ids, transform=train_transforms)
    val_set = MRIGenerationLoader(data_root, val_ids, transform=val_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # These might change based on the params
    model = ViTAutoEnc(
        in_channels=4,
        out_channels=1,
        patch_size=(16, 16, 16),
        spatial_dims=3,
        img_size=(32, 256, 240),
        proj_type='conv',
        dropout_rate=dropout_rate
    )

    #CHANGE PATH HERE
    if os.path.exists('./best_ViTVAE.pt'):
        state = torch.load('./best_ViTVAE.pt', map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=False)

        model.patch_embedding = PatchEmbeddingBlock(
            in_channels=4,
            img_size=(32, 256, 240),
            patch_size=(16, 16, 16),
            hidden_size=768, 
            num_heads=12,
            proj_type='conv',
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
        conv_trans = Conv[Conv.CONVTRANS, model.spatial_dims]
        up_kernel_size = [int(math.sqrt(i)) for i in model.patch_size]
        model.conv3d_transpose_1 = conv_trans(
            in_channels=16, out_channels=1, kernel_size=up_kernel_size, stride=up_kernel_size
        )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()

    for epoch in range(500):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs, _ = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)

                outputs, _ = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                break

    model.load_state_dict(best_model_wts)
    return best_val_loss

# === Run Optuna Study ===
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar = True)

    with open("optuna_prediction_tuning.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "params", "val_loss"])
        for t in study.trials:
            writer.writerow([t.number, t.params, t.value])
