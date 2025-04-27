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
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_viz(save_dir, count, original, scans, preds):
    os.makedirs(save_dir, exist_ok=True)

    mid_slice = original.shape[2] // 2  # middle depth slice (axis=2 is depth)
    for batch_ind in range(len(original)):
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
        save_path = os.path.join(save_dir, f"sample_{count}_b_{batch_ind}.png")
        cv2.imwrite(save_path, final_image)


# === Fixed Constants ===
data_root = '../data/stripped_3_scans/'
patch_size = (16, 16, 16)
mask_ratio = 0.625
batch_size = 8
fixed_num_heads = 8  # assuming

# === Data Split ===
train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

# === Preprocessing ===
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

# === Loss ===
recon_loss = nn.MSELoss(reduction='mean')
ssim = SSIMLoss(spatial_dims=3, data_range=1.0)


def loss_fn(recon_x, x):
    return 0.5 * recon_loss(recon_x, x) + 0.5 * ssim(recon_x, x)


# === Predefined Trials ===
# predefined_trials_roshan = [
#     {"hidden_size": 256, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.10},
#     {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.15},
#     {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.20},
#     {"hidden_size": 512, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.10},
#     {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 4, "dropout": 0.15}
# ]
predefined_trials_roshan = [
    {"hidden_size": 768, "mlp_dim": 3072, "num_layers": 12, "dropout": 0.20},
    {"hidden_size": 1152, "mlp_dim": 3072, "num_layers": 12, "dropout": 0.20},
    {"hidden_size": 1536, "mlp_dim": 3072, "num_layers": 12, "dropout": 0.20},
]

predefined_trials_sana = [
    {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 8, "dropout": 0.20},
    {"hidden_size": 256, "mlp_dim": 1024, "num_layers": 8, "dropout": 0.10},
    {"hidden_size": 512, "mlp_dim": 2048, "num_layers": 4, "dropout": 0.20},
    {"hidden_size": 256, "mlp_dim": 2048, "num_layers": 8, "dropout": 0.10},
    {"hidden_size": 512, "mlp_dim": 1024, "num_layers": 4, "dropout": 0.15}
]

# Set the trials here -
predefined_trials = predefined_trials_roshan


# === Objective Function for Optuna ===
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

    test_viz_dir = os.path.join('./test_viz/', trial_name)

    print(f"\n[Trial {trial_id}] {trial_name} STARTED")

    # Load Datasets
    test_set = MRIDataLoader(data_root, test_ids, transform=val_transforms,
                             mask_scan='random', mask_ratio=mask_ratio)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

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

    os.makedirs("models_optuna", exist_ok=True)
    model_path = os.path.join("models_optuna", f"{trial_name}.pt")
    model.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Evaluate
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            scans = batch["numpy"].to(device)
            original = batch["original"].to(device)
            recon, _ = model(scans)

            make_viz(test_viz_dir, count, original, scans, recon)
            count += 1


# === Main Optuna Execution ===
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=len(
        predefined_trials), show_progress_bar=True)

    print("✅ All trials completed.")
