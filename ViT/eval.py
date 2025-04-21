import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor, Resize
import lpips
from piq import ssim, multi_scale_ssim
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

import sys
sys.path.append('../')
from data_loader_skullstrip import MRISliceDataLoader, MRIGenerationLoader, split_data
from models import ScanOrderViT, TemporalScanPredictor
from eval_utils import calculate_mmd, calculate_coverage
from train_generator import parse_model_config_from_filename

transform = Compose([
    ScaleIntensity(),
    Resize((224, 224)),
    ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load test sets ===
data_root_ssl = '../stripped_3_scans/'
data_root_gen = '../stripped_5_scans/'

_, test_ids_ssl, _ = split_data(os.listdir(data_root_ssl))
_, test_ids_gen, _ = split_data(os.listdir(data_root_gen))

test_set_ssl = MRISliceDataLoader(data_root_ssl, test_ids_ssl, transform=transform)
test_loader_ssl = DataLoader(test_set_ssl, batch_size=4, shuffle=False)

test_set_gen = MRIGenerationLoader(data_root_gen, test_ids_gen, transform=transform)
test_loader_gen = DataLoader(test_set_gen, batch_size=4, shuffle=False)

# === Load models ===
model_path = "best_model_dim128_depth4_heads8_mlp256.pth"
dim, depth, heads, mlp_dim = parse_model_config_from_filename(model_path)
    
encoder = ScanOrderViT(
        image_size=224,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )
encoder.load_state_dict(torch.load(model_path))
encoder.eval()

# --- Evaluate SSL classifier ---
ssl_criterion = torch.nn.CrossEntropyLoss()
total_loss = 0
num_batches = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader_ssl:
        scans = batch["numpy"].to(device)
        labels = batch["label"].to(device)

        logits = encoder.classify(scans)
        loss = ssl_criterion(logits, labels)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        num_batches += 1

avg_test_loss = total_loss / num_batches
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(all_labels)
print(all_preds)

print("\n🧪 SSL Task (Test Set)")
print(f"Loss: {avg_test_loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# === Load generator model ===
generator = TemporalScanPredictor(encoder, dim = dim).to(device)


generator.load_state_dict(torch.load("gen_model_lr4e-04_bs4.pth"))
generator.eval()

# === Evaluate generator ===
def evaluate_and_visualize(model, dataloader):
    model.eval()
    total_ssim, total_msssim, total_lpips = 0, 0, 0
    total_mmd, total_cov = 0, 0
    num_batches = 0

    all_real = []
    all_pred = []

    lpips_metric = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch["input"].to(device)
            target = batch["target"].to(device)

            pred = model(input_seq).clamp(0, 1)

            B = pred.size(0)
            for i in range(B):
                real_img = target[i].squeeze().cpu().numpy()
                pred_img = pred[i].squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(real_img, cmap='gray')
                axes[0].set_title("Real Scan")
                axes[1].imshow(pred_img, cmap='gray')
                axes[1].set_title("Predicted Scan")
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

            total_ssim += ssim(pred, target, data_range=1.0).item()
            total_msssim += multi_scale_ssim(pred, target, data_range=1.0).item()
            total_lpips += lpips_metric(pred, target).mean().item()

            all_real.append(target.view(B, -1).cpu())
            all_pred.append(pred.view(B, -1).cpu())
            num_batches += 1

    all_real_flat = torch.cat(all_real, dim=0)
    all_pred_flat = torch.cat(all_pred, dim=0)

    total_mmd = calculate_mmd(all_real_flat, all_pred_flat)
    total_cov = calculate_coverage(all_real_flat.numpy(), all_pred_flat.numpy())

    print(f"\n\U0001F4CA Generator Metrics Across {num_batches} Batches")
    print(f"MS-SSIM: {total_msssim / num_batches:.4f}")
    print(f"SSIM:    {total_ssim / num_batches:.4f}")
    print(f"LPIPS:   {total_lpips / num_batches:.4f}")
    print(f"MMD:     {total_mmd:.4f}")
    print(f"Coverage:{total_cov:.4f}")

# === Run generator evaluation ===
evaluate_and_visualize(generator, test_loader_gen)
