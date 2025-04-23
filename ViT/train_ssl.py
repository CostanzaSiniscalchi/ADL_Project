import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor, Resize
import copy
from tqdm import tqdm
import pandas as pd
import itertools


from vit_pytorch import ViT
from models import ScanOrderViT

import sys
sys.path.append('../')
from data_loader_skullstrip import MRISliceDataLoader, split_data



def train_ssl(model, dataloader, val_dataloader, optimizer, criterion, epochs=50, patience=5, scheduler= None):
    model.train()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            scans = batch["numpy"].to(device)
            labels = batch["label"].to(device)

            logits = model.classify(scans)
            loss = criterion(logits, labels)

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
                scans = batch["numpy"].to(device)
                labels = batch["label"].to(device)

                logits = model.classify(scans)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"[SSL] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Early stopping logic
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
    
def objective(trial):
    # dim = trial.suggest_categorical('dim', [128, 256, 512])
    # depth = trial.suggest_categorical('depth', [4, 6, 8])
    # heads = trial.suggest_categorical('heads', [4, 8])
    # mlp_dim = trial.suggest_categorical('mlp_dim', [256, 512, 1024])

    combo = trial.suggest_categorical("arch_combo", my_arch_space)
    dim, depth, heads, mlp_dim = combo
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1])

    print(f"\n📌 Trial {trial.number} Hyperparams:")
    print(f"  Architecture: dim={dim}, depth={depth}, heads={heads}, mlp_dim={mlp_dim}")
    print(f"  Training:     lr={lr:.2e}, weight_decay={weight_decay:.1e}, batch_size={batch_size}")
    

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    
    model = ScanOrderViT(
    image_size=image_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim
    ).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    model = train_ssl(model, train_loader, val_loader, optimizer, criterion, epochs=50, patience=5, scheduler = scheduler)

    model_path = f"best_model_dim{dim}_depth{depth}_heads{heads}_mlp{mlp_dim}_lr{lr:.0e}_bs{batch_size}_wd{weight_decay:.0e}.pth"
    torch.save(model.state_dict(), model_path)

    return evaluate_val_loss(model, val_loader, criterion)

def evaluate_val_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            scans = batch["numpy"].to(device)
            labels = batch["label"].to(device)
            logits = model.classify(scans)
            loss = criterion(logits, labels)
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

  
    
    data_root = '../stripped_3_scans_slices/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))
    train_set = MRISliceDataLoader(data_root, train_ids, transform=transform)
    val_set = MRISliceDataLoader(data_root, val_ids, transform=transform)
    
    
    arch_combos = list(itertools.product(
        [128, 256, 512],     # dim
        [4, 6, 8],           # depth
        [4, 8],              # heads
        [256, 512, 1024]     # mlp_dim
    )) # 54 combinations
    
    # Assign subsets to each groupmate
    group_id = 0  # change this to 0, 1, or 2
    split_size = len(arch_combos) // 3
    my_arch_space = arch_combos[group_id * split_size : (group_id + 1) * split_size]


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25, show_progress_bar=True) # 18 different combinations per person (not including LR, Weight decay, BS)
    
    print("Best hyperparameters:", study.best_trial.params)
    
    df = study.trials_dataframe()
    df.to_csv("ssl_trials.csv", index=False)
    df.to_json("ssl_trials.json", orient="records", indent=2)
    
    print("📁 Trials saved to: ssl_trials.csv & ssl_trials.json")
