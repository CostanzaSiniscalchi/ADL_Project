import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from data_loader_ssl import MRIDataLoader, MRIGenerationLoader, split_data, MRISliceDataLoader, MRISliceGeneratorDataLoader
from monai.networks.nets import ViTAutoEnc
from monai.transforms import Compose, ScaleIntensity, ToTensor, Resize
from Autoencoder.eval.eval_utils import calculate_lpips, calculate_mmd, calculate_coverage, calculate_ms_ssim, calculate_ssim
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers import Conv
import math


def make_viz_ssl(epoch, save_dir, count, original, scans, preds):
    save_viz_dir = os.path.join(save_dir, 'viz_test_ssl/')
    os.makedirs(save_viz_dir, exist_ok=True)

    batch_ind = 0

    rows = []
    for s in range(3):  # For each of the 3 scans
        orig_slice = original[batch_ind, s].cpu().numpy()
        scan_slice = scans[batch_ind, s].cpu().numpy()
        pred_slice = preds[batch_ind, s].cpu().detach().numpy()

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
    save_path = os.path.join(save_viz_dir, f"epoch_{epoch}_sample_{count}.png")
    cv2.imwrite(save_path, final_image)


def make_viz_pred(epoch, save_dir, count, targets, inputs, preds):
    save_viz_dir = os.path.join(save_dir, 'viz_val_pred/')
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


# def make_viz(save_dir, count, original, scans, preds):
#     os.makedirs(save_dir, exist_ok=True)
#     batch_ind = 0
#     mid_slice = original.shape[2] // 2

#     rows = []
#     for s in range(scans.shape[1]):
#         orig_slice = original[batch_ind, s, mid_slice].cpu().numpy()
#         scan_slice = scans[batch_ind, s, mid_slice].cpu().numpy()
#         pred_slice = preds[batch_ind, s, mid_slice].cpu().detach().numpy()

#         def normalize(img):
#             img = np.clip(img, 0.0, 1.0)
#             return (img * 255).astype(np.uint8)

#         row = np.concatenate([
#             normalize(scan_slice),
#             normalize(orig_slice),
#             normalize(pred_slice)
#         ], axis=1)
#         rows.append(row)

#     final_image = np.concatenate(rows, axis=0)
#     save_path = os.path.join(save_dir, f"sample_{count}.png")
#     cv2.imwrite(save_path, final_image)

def test_ssl(model, test_dataloader, device):
    model.to(device)
    model.eval()
    count = 0
    mse_loss = nn.MSELoss()
    total_mse = 0
    total_ssim = 0
    total_msssim = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="SSL Eval"):
            scans = batch["numpy"].to(device)
            original = batch["original"].to(device)
            recon_batch, _ = model(scans)
            recon_batch = torch.clamp(recon_batch, 0.0, 1.0)
            original = torch.clamp(original, 0.0, 1.0)

            loss = mse_loss(recon_batch, original)
            total_mse += loss.item()

            total_ssim += calculate_ssim(recon_batch, original)
            total_msssim += calculate_ms_ssim(recon_batch, original)

            make_viz_ssl(0, './test_viz', count, original, scans, recon_batch)
            
            count += 1
            num_batches += 1

    print(f"[SSL] MSE Loss:   {total_mse / num_batches:.4f}")
    print(f"[SSL] SSIM:       {total_ssim / num_batches:.4f}")
    print(f"[SSL] MS-SSIM:    {total_msssim / num_batches:.4f}")


def test_prediction(model, test_dataloader, device):
    model.to(device)
    model.eval()
    total_msssim = 0
    total_ssim = 0
    total_lpips = 0
    total_mmd = 0
    total_covered_samples = 0
    total_generated_samples =0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Prediction Eval"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds, _ = model(inputs)

            # Pass through decoder layers
            preds = model.relu(model.decoder_conv1(preds))
            preds = model.relu(model.decoder_conv2(preds))
            preds = model.decoder_conv3(preds)

            preds = torch.clamp(preds, 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)

            total_msssim += calculate_ms_ssim(preds, targets)
            total_ssim += calculate_ssim(preds, targets)
            total_lpips += calculate_lpips(preds, targets)
            #total_mmd += calculate_mmd(preds, targets)

            flat_preds = preds.view(preds.size(0), -1)    # (batch_size, flattened_feature_dim)
            flat_targets = targets.view(targets.size(0), -1)
            total_cov, batch_total = calculate_coverage(flat_preds.cpu(), flat_targets.cpu())
            total_covered_samples += total_cov
            total_generated_samples += batch_total
            num_batches +=1

            make_viz_pred(0,'./test_viz', num_batches, targets, inputs, preds)

    print(f"MS-SSIM: {total_msssim / num_batches:.4f}")
    print(f"SSIM:    {total_ssim / num_batches:.4f}")
    print(f"LPIPS:   {total_lpips / num_batches:.4f}")
    print(f"MMD:     {total_mmd:.4f}")
    print(f"Coverage: {total_covered_samples / total_generated_samples:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = Compose([
    lambda x: x[np.newaxis, ...],  # (1, H, W)
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((224, 224)),
    ToTensor(),
    lambda x: x.squeeze(0)
    ])

    data_root_ssl = '../../data/stripped_3_scans_slices/'
    train_ids, test_ids, val_ids = split_data(os.listdir(data_root_ssl))


    test_set_ssl = MRISliceDataLoader(data_root_ssl, test_ids,
                                   transform=test_transforms, mask_scan='random', mask_ratio='random')
    test_loader_ssl = DataLoader(test_set_ssl, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model_ssl = ViTAutoEnc(
    in_channels=3,
    out_channels=3,
    patch_size=(16, 16),
    spatial_dims=2,
    img_size=(224, 224),
    proj_type='conv',
    dropout_rate=0.2,
    hidden_size=768,
    mlp_dim=3072,
    )

    state = torch.load('best_768_3072.pt', map_location=device)
    model_ssl.load_state_dict(state['model_state_dict'])

    #test_ssl(model_ssl, test_loader_ssl, device)

    # Prediction task testing
    data_root_pred = '../../data/stripped_5_scans_slices/'


    test_set_pred = MRISliceGeneratorDataLoader(data_root_pred, test_ids, transform=test_transforms, mask_scan='random', mask_ratio='random')
  
    test_loader_pred = DataLoader(test_set_pred, batch_size=8,
                              shuffle=True, num_workers=4, pin_memory=True)

    model_pred = ViTAutoEnc(in_channels=3, out_channels=3, patch_size=(16, 16), spatial_dims=2,
                       img_size=(224, 224), proj_type='conv', dropout_rate=0.2, hidden_size=768, mlp_dim=3072)

    
    model_pred.patch_embedding = PatchEmbeddingBlock(
        in_channels=4,
        img_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        proj_type='conv',
        dropout_rate=0.2,
        spatial_dims=2,
        hidden_size=768,
    )
    conv_trans = Conv[Conv.CONVTRANS, model_pred.spatial_dims]
    up_kernel_size = [int(math.sqrt(i)) for i in model_pred.patch_size]
    model_pred.conv3d_transpose_1 = conv_trans(
        in_channels=16, out_channels=1, kernel_size=up_kernel_size, stride=up_kernel_size
    )

    model_pred.decoder_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    model_pred.decoder_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
    model_pred.decoder_conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    model_pred.relu = nn.ReLU()

    model_pred.to(device)

    
    
    state = torch.load('training_runs_conv/pred_vitvae_768_3072/best_768_3072.pt', map_location=device, weights_only=True)
    model_pred.load_state_dict(state['model_state_dict'], strict=False)
    
    model_pred.to(device)
    
    test_prediction(model_pred, test_loader_pred, device)








