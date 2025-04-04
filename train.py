# Define a transformation pipeline for MRI data
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import AutoencoderKL
import torch
from data_loader import MRIDataLoader, split_data
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor
import numpy as np
import os

# train test split
np_dir = 'data/numpy_data_t1w_3_AD_scans'
ids = os.listdir(np_dir)
print(ids)
print(len(ids))
# Example usage:
train, test, validation = split_data(ids)

print("Train IDs:", train)
print("Test IDs:", test)
print("Validation IDs:", validation)

transform = Compose([ScaleIntensity(), ToTensor()])

# Initialize the dataset loader
train_mri_loader = MRIDataLoader(
    numpy_dir=np_dir, id_list=train, transform=transform)
test_mri_loader = MRIDataLoader(
    numpy_dir=np_dir, id_list=test, transform=transform)
val_mri_loader = MRIDataLoader(
    numpy_dir=np_dir, id_list=validation, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_mri_loader, batch_size=1, shuffle=True)
test_loader = DataLoader(test_mri_loader, batch_size=1, shuffle=True)
val_loader = DataLoader(val_mri_loader, batch_size=1, shuffle=True)


print(len(train_mri_loader))
sample = next(iter(train_loader))  # check if iterating works
# Check the shape of the data
print("Sample data shape:", sample['numpy'].shape)

# Sample label
print("Sample labels:", sample['labels'])


# Train function
def train_model(train_loader, model, encoder, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Shape: (batch_size, num_scans, channels, D, H, W)
        inputs = batch["numpy"].to(device)
        labels = batch["labels"].to(device)  # Shape: (batch_size, num_scans)

        batch_loss = 0
        for i in range(inputs.size(1)):  # Loop over each scan in the batch
            # Get the latent representation for the current scan
            # Shape: (batch_size, channels, D, H, W)
            scan_input = inputs[:, i:i+1, :, :, :]
            print(f"Shape of scan_input {i}: {scan_input.shape}")
            # Pass the scan through the encoder
            latent_rep = encoder(scan_input)  # Get the encoded features
            print(f"Latent representation shape: {latent_rep.shape}")

            # Flatten the latent representation
            # Flatten to (batch_size, latent_dim)
            latent_rep = latent_rep.view(latent_rep.size(0), -1)

            # Forward pass through the TemporalOrderModel
            optimizer.zero_grad()
            outputs = model(latent_rep)

            # Compute the loss for the current scan
            loss = criterion(outputs, labels[:, i])
            loss.backward()

            # Add this loss to the batch loss
            batch_loss += loss.item()

        # Average loss over the batch
        optimizer.step()  # Apply gradients after processing all scans in the batch
        total_loss += batch_loss / inputs.size(1)  # Average loss per scan

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Validation model


def validate_model(val_loader, model, encoder, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["numpy"].to(device)
            labels = batch["labels"].to(device)

            batch_loss = 0
            for i in range(inputs.size(1)):  # Loop over each scan in the batch
                scan_input = inputs[:, i, :, :, :]
                latent_rep = encoder(scan_input)  # Get the latent features

                # Flatten the latent representation
                latent_rep = latent_rep.view(latent_rep.size(0), -1)

                # Forward pass through TemporalOrderModel
                outputs = model(latent_rep)

                # Compute the loss
                loss = criterion(outputs, labels[:, i])
                batch_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels[:, i]).sum().item()

            total_loss += batch_loss / inputs.size(1)  # Average loss per scan

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# Define the TemporalOrderModel class

class TemporalOrderModel(nn.Module):
    def __init__(self, input_size):
        super(TemporalOrderModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output 3 classes for 3 possible orders

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

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

input_size = 8 * 44 * 64 * 60
temporal_order_model = TemporalOrderModel(input_size=input_size)

# Move models to the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
temporal_order_model.to(device)
autoencoder_model.to(device)

# Define optimizer, using both temporal_order_model and autoencoder_model (if fine-tuning the encoder)
optimizer = optim.Adam(
    list(temporal_order_model.parameters()) +
    list(autoencoder_model.parameters()),
    lr=0.001
)

# Check model architecture for debugging
# print("TemporalOrderModel architecture:", temporal_order_model)
# print("Autoencoder architecture:", autoencoder_model)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(train_loader, temporal_order_model,
                             autoencoder_model.encoder, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_accuracy = validate_model(
        val_loader, temporal_order_model, autoencoder_model.encoder, criterion, device)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
