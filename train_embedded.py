# Define a transformation pipeline for MRI data
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import AutoencoderKL
import torch
from data_loader import EmbeddedMRIDataLoader, split_data
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, ToTensor
import numpy as np
import os

# train test split
np_dir = 'data/embeddings'
ids = os.listdir(np_dir)
print(ids)
print(len(ids))
# Example usage:
train, test, validation = split_data(ids)

print("Train IDs:", train)
print("Test IDs:", test)
print("Validation IDs:", validation)

transform = Compose([ToTensor()])

# Initialize the dataset loader
train_mri_loader = EmbeddedMRIDataLoader(
    numpy_dir=np_dir, id_list=train, transform=transform)
test_mri_loader = EmbeddedMRIDataLoader(
    numpy_dir=np_dir, id_list=test, transform=transform)
val_mri_loader = EmbeddedMRIDataLoader(
    numpy_dir=np_dir, id_list=validation, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_mri_loader, batch_size=8, shuffle=True)
test_loader = DataLoader(test_mri_loader, batch_size=8, shuffle=True)
val_loader = DataLoader(val_mri_loader, batch_size=8, shuffle=True)


print(len(train_mri_loader))
sample = next(iter(train_loader))  # check if iterating works
# Check the shape of the data
print("Sample data shape:", sample['numpy'].shape)

# Sample label
print("Sample labels:", sample['labels'])


# Train function
def train_model(train_loader, model, criterion, optimizer, device):
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

            # Flatten the latent representation
            # Flatten to (batch_size, latent_dim)
            scan_input = scan_input.view(scan_input.size(0), -1)

            # Forward pass through the TemporalOrderModel
            optimizer.zero_grad()
            outputs = model(scan_input)

            print(labels, outputs)

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


def validate_model(val_loader, model, criterion, device):
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

                # Flatten the latent representation
                scan_input = scan_input.view(scan_input.size(0), -1)

                # Forward pass through TemporalOrderModel
                outputs = model(scan_input)

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

input_size = 8 * 44 * 64 * 60
temporal_order_model = TemporalOrderModel(input_size=input_size)

# Move models to the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
temporal_order_model.to(device)

# Define optimizer, using both temporal_order_model and autoencoder_model (if fine-tuning the encoder)
optimizer = optim.Adam(
    temporal_order_model.parameters(),
    lr=0.001
)

# Check model architecture for debugging
# print("TemporalOrderModel architecture:", temporal_order_model)
# print("Autoencoder architecture:", autoencoder_model)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(
        train_loader, temporal_order_model, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_accuracy = validate_model(
        val_loader, temporal_order_model, criterion, device)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
