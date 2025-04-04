# Define a transformation pipeline for MRI data
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import AutoencoderKL
import torch
from data_loader import AllMRIDataLoader, split_data
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor
import numpy as np
import os

# train test split
np_dir = 'data/numpy_data_t1w_3_AD_scans'
ids = os.listdir(np_dir)
print(ids)
print(len(ids))

transform = Compose([ScaleIntensity(), ToTensor()])

mri_loader = AllMRIDataLoader(
    numpy_dir=np_dir, id_list=ids, transform=transform)
loader = DataLoader(mri_loader, batch_size=1, shuffle=True)


print(len(loader))
sample = next(iter(loader))  # check if iterating works
# Check the shape of the data
print("Sample data shape:", sample['numpy'].shape)

# Sample label
print("Sample file_names:", sample['file_names'])


def make_embeddings(train_loader, encoder, device, save_dir="data/embeddings"):
    encoder.eval()  # Make sure encoder is in eval mode
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  # Disable gradient tracking for efficiency
        for batch in train_loader:
            inputs = batch["numpy"].to(device)  # (B, num_scans, C, D, H, W)
            file_names = batch["file_names"]  # list of lists
            patient_id = batch["patient_id"][0]

            print(f"Processing patient: {patient_id}")

            patient_folder = os.path.join(save_dir, patient_id)
            os.makedirs(patient_folder, exist_ok=True)

            for i in range(inputs.size(1)):  # Loop over each scan
                scan_input = inputs[:, i:i+1, :, :, :]  # (B, 1, C, D, H, W)
                print(f"Shape of scan_input {i}: {scan_input.shape}")

                # Get latent representation
                latent_rep = encoder(scan_input)
                print(f"Latent shape: {latent_rep.shape}")

                latent_np = latent_rep.squeeze(
                    0).cpu().numpy()  # Remove batch dim

                original_filename = os.path.basename(file_names[i][0])
                save_path = os.path.join(patient_folder, original_filename)

                np.save(save_path, latent_np)
                print(f"Saved embedding to: {save_path}")


# Initialize the encoder and model
# Define the autoencoder model (make sure it matches the saved model architecture)
autoencoder_model = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=8,
    channels=[64, 128, 256],
    num_res_blocks=2,
    norm_num_groups=32,
    norm_eps=1e-06,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    include_fc=False
)

# Load the pretrained weights
autoencoder_path = './models/model_autoencoder.pt'
state_dict = torch.load(autoencoder_path)
# Load the state dict with strict=False to ignore mismatched keys
autoencoder_model.load_state_dict(state_dict, strict=False)
# Set the model to evaluation mode
autoencoder_model.eval()

# Move models to the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
autoencoder_model.to(device)


make_embeddings(loader, autoencoder_model.encoder, device)
