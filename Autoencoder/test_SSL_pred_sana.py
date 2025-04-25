import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
from data_loader_ssl import MRIDataLoader, MRIGenerationLoader, split_data
from monai.networks.nets import ViTAutoEnc
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
import cv2
import sys
from eval_utils import compute_lpips, compute_mmd, compute_coverage, compute_msssim, compute_ssim  # Assumed utility funcs

def make_viz(save_dir, count, original, scans, preds):
    os.makedirs(save_dir, exist_ok=True)

    batch_ind = 0
    mid_slice = original.shape[2] // 2

    rows = []
    for s in range(3):
        orig_slice = original[batch_ind, s, mid_slice].cpu().numpy()
        scan_slice = scans[batch_ind, s, mid_slice].cpu().numpy()
        pred_slice = preds[batch_ind, s, mid_slice].cpu().detach().numpy()

        def normalize(img):
            img = np.clip(img, 0.0, 1.0)
            return (img * 255).astype(np.uint8)

        row = np.concatenate([
            normalize(scan_slice),
            normalize(orig_slice),
            normalize(pred_slice)
        ], axis=1)
        rows.append(row)

    final_image = np.concatenate(rows, axis=0)
    save_path = os.path.join(save_dir, f"sample_{count}.png")
    cv2.imwrite(save_path, final_image)

def test_ssl(model, test_dataloader):
    model.to(device)
    model.eval()
    count = 0
    mse_loss = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="SSL Eval"):
            scans = batch["numpy"].to(device)
            original = batch["original"].to(device)
            recon_batch, _ = model(scans)
            loss = mse_loss(recon_batch, original)
            total_loss += loss.item()
            make_viz('./testing_viz_vit_do_nosig_random/', count, original, scans, recon_batch)
            count += 1
    avg_loss = total_loss / len(test_dataloader)
    print(f"[SSL] MSE Loss: {avg_loss:.4f}")

def test_prediction(model, test_dataloader):
    model.to(device)
    model.eval()
    total_msssim = 0
    total_ssim = 0
    total_lpips = 0
    total_mmd = 0
    total_cov = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Prediction Eval"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds, _ = model(inputs)

            total_msssim += compute_msssim(preds, targets)
            total_ssim += compute_ssim(preds, targets)
            total_lpips += compute_lpips(preds, targets)
            total_mmd += compute_mmd(preds, targets)
            total_cov += compute_coverage(preds, targets)
            num_batches += 1

    print(f"MS-SSIM: {total_msssim / num_batches:.4f}")
    print(f"SSIM:    {total_ssim / num_batches:.4f}")
    print(f"LPIPS:   {total_lpips / num_batches:.4f}")
    print(f"MMD:     {total_mmd:.4f}")
    print(f"Coverage:{total_cov:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor()
    ])

    data_root = '../data/stripped_3_scans/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

    test_set_ssl = MRIDataLoader(data_root, test_ids, transform=test_transforms, mask_scan='random')
    test_loader_ssl = DataLoader(test_set_ssl, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model_ssl = ViTAutoEnc(in_channels=3, out_channels=3, patch_size=(16, 16, 16), spatial_dims=3,
                           img_size=(32, 256, 240), proj_type='conv', dropout_rate=0.2)
    model_ssl.load_state_dict(torch.load('./best_ViTVAE.pt', weights_only=True)['model_state_dict'])
    test_ssl(model_ssl, test_loader_ssl)

    # Prediction task testing
    data_root_pred = '../data/stripped_5_scans/'
    test_set_pred = MRIGenerationLoader(data_root_pred, test_ids, transform=test_transforms)
    test_loader_pred = DataLoader(test_set_pred, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model_pred = ViTAutoEnc(in_channels=4, out_channels=1, patch_size=(16, 16, 16), spatial_dims=3,
                            img_size=(32, 256, 240), proj_type='conv', dropout_rate=0.2)
    model_pred.load_state_dict(torch.load('./best_ViTVAE_pred.pt', map_location=device)['model_state_dict'])
    test_prediction(model_pred, test_loader_pred)
