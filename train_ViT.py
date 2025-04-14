# Define a transformation pipeline for MRI data
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import ViT
import torch
from data_loader import MRISliceDataLoader, split_data
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
train_mri_loader = MRISliceDataLoader(
    numpy_dir=np_dir, id_list=train, transform=transform)
test_mri_loader = MRISliceDataLoader(
    numpy_dir=np_dir, id_list=test, transform=transform)
val_mri_loader = MRISliceDataLoader(
    numpy_dir=np_dir, id_list=validation, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_mri_loader, batch_size=4, shuffle=True)
test_loader = DataLoader(test_mri_loader, batch_size=4, shuffle=True)
val_loader = DataLoader(val_mri_loader, batch_size=4, shuffle=True)

# Train function


def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Shape: (batch_size, num_scans, channels, D, H, W)
        inputs = batch["numpy"].to(device)
        labels = batch["label"].to(device)  # Shape: (batch_size, num_scans)

        # Forward pass through the TemporalOrderModel
        optimizer.zero_grad()
        outputs, hidden_states = model(inputs)

        print(outputs)
        print(labels)

        # Compute the loss for the current scan
        loss = criterion(outputs, labels)
        loss.backward()

        # Average loss over the batch
        optimizer.step()  # Apply gradients after processing all scans in the batch
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Validation model


def validate_model(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["numpy"].to(device)
            labels = batch["label"].to(device)

            outputs, hidden_states = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Get predictions
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


print(len(train_mri_loader))
sample = next(iter(train_loader))  # check if iterating works

img_size = sample['numpy'].shape[-2:]
# Check the shape of the data
print("Sample data shape:", sample['numpy'].shape, " image size: ", img_size)

# Sample label
print("Sample labels:", sample['label'])

ViT_net = ViT(in_channels=3, img_size=img_size, patch_size=16, spatial_dims=2, num_classes=6,
              proj_type='conv', pos_embed_type='sincos', classification=True, post_activation="no_need")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Move models to the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ViT_net.to(device)

# Define optimizer, using both temporal_order_model and autoencoder_model (if fine-tuning the encoder)
optimizer = optim.Adam(ViT_net.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(train_loader, ViT_net,
                             criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_accuracy = validate_model(
        val_loader, ViT_net, criterion, device)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
