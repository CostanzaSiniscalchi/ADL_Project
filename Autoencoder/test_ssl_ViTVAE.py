import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
from data_loader_ssl import MRIDataLoader, split_data
from monai.networks.nets import ViTAutoEnc
from monai.transforms import RandGaussianNoise, RandAffine, Compose, ScaleIntensity, ToTensor
from monai.losses import SSIMLoss
import cv2
import sys


def make_viz(save_dir, count, original, scans, preds):
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
    save_path = os.path.join(save_dir, f"sample_{count}.png")
    cv2.imwrite(save_path, final_image)


def test_ssl(model, test_dataloader):
    model.to(device)
    model.eval()
    count = 0
    for batch in tqdm(test_dataloader, desc=f"Count {count}"):
        scans = batch["numpy"].to(device)        # (B, 3, 32, 256, 240)
        original = batch["original"].to(device)  # (B, 3, 32, 256, 240)

        recon_batch, hidden_states = model(scans)

        make_viz('./testing_viz_vit_do_nosig_random/',
                 count, original, scans, recon_batch)
        count += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor()
    ])

    data_root = '../data/stripped_3_scans/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root))

    test_set = MRIDataLoader(
        data_root, test_ids, transform=test_transforms, mask_scan='random')

    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, num_workers=4, pin_memory=True)

    model = ViTAutoEnc(in_channels=3, out_channels=3, patch_size=(16, 16, 16), spatial_dims=3,
                       img_size=(32, 256, 240), proj_type='conv', dropout_rate=0.2)

    model.load_state_dict(torch.load('./best_ViTVAE.pt',
                          weights_only=True)['model_state_dict'])

    trained_model = test_ssl(model, test_loader)
