"""
Filename: train_ssl_slice_tuning.py
Author: Roshan Kenia & Sanmati Choudhary
Description: Hyperparameter tuning for SSL task using Optuna and ViTAutoEnc model.
"""


import os
import time
import csv
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
from data_loader_ssl import MRISliceDataLoader, split_data
from model import ViTAutoEnc
from Autoencoder.ssl.train_ssl_slice import train_ssl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = '../../data/stripped_3_scans_slices/'
patch_size = (16, 16, 16)
mask_ratio = 0.625
batch_size = 8
fixed_num_heads = 8  

# Splitting Data
train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

# Preprocessing 
train_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    RandGaussianNoise(prob=0.2, std=0.01),
    RandAffine(prob=0.3, rotate_range=(0.05, 0.05, 0.05)),
    ToTensor()
])

val_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    ToTensor()
])

# Loss defintions
recon_loss = nn.MSELoss(reduction='mean')
ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

def loss_fn(recon_x, x):
    return 0.5 * recon_loss(recon_x, x) + 0.5 * ssim(recon_x, x)

# Predefined Trials 
predefined_trials_roshan = [
    {"hidden_size": 256, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.10},
    {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.15},
    {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.20},
    {"hidden_size": 512, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.10},
    {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 4, "dropout": 0.15}
]

predefined_trials_sana = [
    {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 8, "dropout": 0.20},
    {"hidden_size": 256, "mlp_dim": 1024, "num_layers": 8, "dropout": 0.10},
    {"hidden_size": 512, "mlp_dim": 2048, "num_layers": 4, "dropout": 0.20},
    {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.10},
    {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.15}
]

#Set the trials here - 
predefined_trials = predefined_trials_roshan

os.makedirs("optuna_results", exist_ok=True)
result_file = f"optuna_results/optuna_results_fixed_2.csv"

with open(result_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "params", "val_loss"])

# Objective Function for Optuna 
def objective(trial):
    trial_id = trial.number
    if trial_id >= len(predefined_trials):
        raise optuna.exceptions.TrialPruned()

    params = predefined_trials[trial_id]
    hidden_size = params['hidden_size']
    mlp_dim = params['mlp_dim']
    num_layers = params['num_layers']
    dropout = params['dropout']

    trial_name = f"ViTAE_{trial_id}_h{hidden_size}_m{mlp_dim}_l{num_layers}_d{int(dropout*10)}"

    print(f"\n[Trial {trial_id}] {trial_name} STARTED")

    # Load Datasets
    train_set = MRISliceDataLoader(data_root, train_ids, transform=train_transforms, mask_scan='random', mask_ratio=mask_ratio)
    val_set = MRISliceDataLoader(data_root, val_ids, transform=val_transforms, mask_scan='random', mask_ratio=mask_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = ViTAutoEnc(
        in_channels=3,
        out_channels=3,
        patch_size=patch_size,
        spatial_dims=3,
        img_size=(32, 256, 240),
        proj_type='conv',
        dropout_rate=dropout,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=fixed_num_heads,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    model = train_ssl(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        epochs=400,
        patience=15,
        trial_name=trial_name)

    # Save Model
    os.makedirs("models_optuna", exist_ok=True)
    model_path = os.path.join("models_optuna", f"{trial_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Trial {trial_id}] DONE — model saved at {model_path}")

    # Evaluate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            scans = batch["numpy"].to(device)
            original = batch["original"].to(device)
            recon, _ = model(scans)
            val_loss += loss_fn(recon, original).item()

    avg_val_loss = val_loss / len(val_loader)

    # IMMEDIATE SAVE after trial incase a trial fails
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial_id, params, avg_val_loss])

    print(f"[Trial {trial_id}] Logged result — val_loss={avg_val_loss:.6f}")

    return avg_val_loss

# Main Optuna Execution
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=len(predefined_trials), show_progress_bar=True)

    print("✅ All trials completed.")
