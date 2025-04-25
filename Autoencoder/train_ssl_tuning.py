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
from data_loader_ssl import MRIDataLoader, split_data
from model import ViTAutoEnc
from train_ssl_ViTVAE import train_ssl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = '../data/stripped_3_scans/'
train_ids, test_ids, val_ids = split_data(os.listdir(data_root))
val_transforms = Compose([ScaleIntensity(minv=0.0, maxv=1.0), ToTensor()])
recon_loss = nn.MSELoss(reduction='mean')
ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

def loss_fn(recon_x, x):
    return 0.5 * recon_loss(recon_x, x) + 0.5 * ssim(recon_x, x)

USER_ID = 1  # 0 or 1

# === Objective Function for Optuna ===
def objective(trial):
    if trial.number % 2 != USER_ID:
        raise optuna.exceptions.TrialPruned()

    patch_size = trial.suggest_categorical("patch_size", [(16, 16, 16)])
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512])
    mlp_dim = trial.suggest_categorical("mlp_dim", [1024, 2048])
    num_layers = trial.suggest_categorical("num_layers", [4, 8])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.15, 0.2])
    batch_size = trial.suggest_categorical("batch_size", [8])
    mask_ratio = trial.suggest_categorical("mask_ratio", [0.5, 0.625, 0.75])

    trial_name = f"ViTAE_{trial.number}_p{patch_size[0]}_h{hidden_size}_m{mlp_dim}_l{num_layers}_d{int(dropout*10)}_bs{batch_size}_mr{int(mask_ratio*100)}"
    print(f"\n[Trial {trial.number}] {trial_name} STARTED")

    train_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandGaussianNoise(prob=0.2, std=0.01),
        RandAffine(prob=0.3, rotate_range=(0.05, 0.05, 0.05)),
        ToTensor()
    ])
    train_set = MRIDataLoader(data_root, train_ids, transform=train_transforms, mask_scan='random', mask_ratio=mask_ratio)
    val_set = MRIDataLoader(data_root, val_ids, transform=val_transforms, mask_scan='random', mask_ratio=mask_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ViTAutoEnc(
        in_channels=3, out_channels=3, patch_size=patch_size, spatial_dims=3,
        img_size=(32, 256, 240), proj_type='conv', dropout_rate=dropout,
        hidden_size=hidden_size, mlp_dim=mlp_dim, num_layers=num_layers, num_heads=8
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model = train_ssl(model, train_loader, val_loader, optimizer, loss_fn, epochs=500, patience=5)

    model_path = os.path.join("models_optuna", f"{trial_name}.pt")
    os.makedirs("models_optuna", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"[Trial {trial.number}] DONE — model saved at {model_path}")

    # Evaluate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            scans = batch["numpy"].to(device)
            original = batch["original"].to(device)
            recon, _ = model(scans)
            val_loss += loss_fn(recon, original).item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# === Run Optuna Study Only ===
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    with open(f"optuna_results_user{USER_ID}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "params", "val_loss"])
        for t in study.trials:
            if t.number % 2 == USER_ID:
                writer.writerow([t.number, t.params, t.value])
