import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor, Resize
import copy
import re

from vit_pytorch import ViT
from models import ScanOrderViT, TemporalScanPredictor
import pandas as pd

import sys
sys.path.append('../')
from data_loader import MRIGenerationLoader, split_data
from tqdm import tqdm



def train_generator(model, dataloader, val_dataloader, optimizer, criterion, epochs=5, patience=5, scheduler = None):

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_seq = batch["input"].to(device)     # [B, 4, 1, H, W]
            target = batch["target"].to(device)       # [B, 1, H, W]

            pred = model(input_seq)                   # → [B, 1, H, W]
            loss = criterion(pred, target)

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
                input_seq = batch["input"].to(device)
                target = batch["target"].to(device)
                pred = model(input_seq)
                loss = criterion(pred, target)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"[GEN] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model

def parse_model_config_from_filename(filename):
    match = re.search(r'dim(\d+)_depth(\d+)_heads(\d+)_mlp(\d+)', filename)
    if not match:
        raise ValueError(f"Could not parse model config from {filename}")
    dim, depth, heads, mlp_dim = map(int, match.groups())
    return dim, depth, heads, mlp_dim

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model_path = "best_model_dim256_depth8_heads4_mlp256.pth"
    dim, depth, heads, mlp_dim = parse_model_config_from_filename(model_path)
    
    encoder = ScanOrderViT(
        image_size=224,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )
    encoder.load_state_dict(torch.load(model_path))
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    model = TemporalScanPredictor(encoder, dim = dim).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
   
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    model = train_generator(model, train_loader, val_loader, optimizer, criterion, epochs=30, patience=5, scheduler=scheduler)

    model_path = f"gen_model_{optimizer_name}_lr{lr:.0e}_bs{batch_size}.pth"
    torch.save(model.state_dict(), model_path)

    return evaluate_val_loss(model, val_loader, criterion)

def evaluate_val_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch["input"].to(device)
            target = batch["target"].to(device)
            pred = model(input_seq)
            loss = criterion(pred, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_size = 224
    transform = Compose([
        ScaleIntensity(),
        Resize((image_size, image_size)),
        ToTensor()
    ])
    
    data_root = '../numpy_conversions_5_AD/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))
    train_set = MRIGenerationLoader(data_root, train_ids, transform=transform)
    val_set = MRIGenerationLoader(data_root, val_ids, transform=transform)
        
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    
    print("Best hyperparameters:", study.best_trial.params)
    
    
    df = study.trials_dataframe()
    df.to_csv("generator_trials.csv", index=False)
    df.to_json("generator_trials.json", orient="records", indent=2)
    
    print("📁 Trials saved to: generator_trials.csv & generator_trials.json")
