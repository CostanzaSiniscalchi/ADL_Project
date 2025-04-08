# Train Test Val Split

# 80 - 10 - 10 split

from math import comb
import torch
from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.data import Dataset, DataLoader
import numpy as np
import random
import os


def split_data(ids, train_size=0.7, test_size=0.15, validation_size=0.15, seed=100):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Shuffle the list of IDs
    random.shuffle(ids)

    # Calculate split indices
    total_size = len(ids)
    train_end = int(train_size * total_size)
    test_end = train_end + int(test_size * total_size)

    # Split the data
    train_ids = ids[:train_end]
    test_ids = ids[train_end:test_end]
    validation_ids = ids[test_end:]

    return train_ids, test_ids, validation_ids


class MRIDataLoader:
    def __init__(self, numpy_dir, id_list, random_order=True, transform=None):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        labels = []
        for patient_id in self.patient_ids:
            # List the scan dates for each patient
            scan_dates = sorted(os.listdir(
                os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for scan_date in scan_dates:
                # Get the corresponding numpy file path
                numpy_file = os.path.join(self.numpy_dir, patient_id, scan_date,
                                          f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy")

                # Check if the numpy file exists
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)  # Append MRI file path
                    # Assign the label based on the scan date
                    if scan_date == 'PREBL00':
                        patient_label.append(0)
                    elif scan_date == 'PREFU12':
                        patient_label.append(1)
                    else:
                        patient_label.append(2)
            all_data.append(patient_data)
            labels.append(patient_label)

        return all_data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numpy_files = self.data[idx]

        label = self.labels[idx]  # Get the corresponding label

        if self.random_order:
            # Shuffle the files and labels together
            combined = list(zip(numpy_files, label))
            random.shuffle(combined)
            numpy_files, label = zip(*combined)  # Unzip back to separate lists

        numpy_data = []
        # Load corresponding numpy file
        for file in numpy_files:
            # Apply transformations if provided
            data = np.load(file)  # Load the scan as a numpy array
            if self.transform:
                data = self.transform(data)  # Apply transformations
            numpy_data.append(data)

        # Stack the data for the sequence of scans
        # Shape will be (num_scans, channels, D, H, W)
        numpy_data = np.stack(numpy_data, axis=0)

        # Return the data as a dictionary
        return {"labels": torch.tensor(label, dtype=torch.long), "numpy": torch.tensor(numpy_data, dtype=torch.float32)}


class EmbeddedMRIDataLoader:
    def __init__(self, numpy_dir, id_list, random_order=True, transform=None):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        labels = []
        for patient_id in self.patient_ids:
            # List the scan dates for each patient
            numpy_files = sorted(os.listdir(
                os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for numpy_file in numpy_files:
                # Append MRI file path
                patient_data.append(os.path.join(
                    self.numpy_dir, patient_id, numpy_file))
                # Assign the label based on the scan date
                if 'PREBL00' in numpy_file:
                    patient_label.append(0)
                elif 'PREFU12' in numpy_file:
                    patient_label.append(1)
                else:
                    patient_label.append(2)
            all_data.append(patient_data)
            labels.append(patient_label)

        return all_data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numpy_files = self.data[idx]

        label = self.labels[idx]  # Get the corresponding label

        if self.random_order:
            # Shuffle the files and labels together
            combined = list(zip(numpy_files, label))
            random.shuffle(combined)
            numpy_files, label = zip(*combined)  # Unzip back to separate lists

        numpy_data = []
        # Load corresponding numpy file
        for file in numpy_files:
            # Apply transformations if provided
            data = np.load(file)  # Load the scan as a numpy array
            if self.transform:
                data = self.transform(data)  # Apply transformations
            numpy_data.append(data)

        # Stack the data for the sequence of scans
        # Shape will be (num_scans, channels, D, H, W)
        numpy_data = np.stack(numpy_data, axis=0)

        # Return the data as a dictionary
        return {"labels": torch.tensor(label, dtype=torch.long), "numpy": torch.tensor(numpy_data, dtype=torch.float32)}


class AllMRIDataLoader:
    def __init__(self, numpy_dir, id_list, transform=None):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.data = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        for patient_id in self.patient_ids:
            # List the scan dates for each patient
            scan_dates = sorted(os.listdir(
                os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            for scan_date in scan_dates:
                # Get the corresponding numpy file path
                numpy_file = os.path.join(self.numpy_dir, patient_id, scan_date,
                                          f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy")

                # Check if the numpy file exists
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)  # Append MRI file path
            patient_dict = {patient_id: patient_data}
            all_data.append(patient_dict)

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_dict = self.data[idx]
        patient_id = list(patient_dict.keys())[0]

        numpy_data = []
        file_names = []
        # Load corresponding numpy file
        for file in patient_dict[patient_id]:
            # Apply transformations if provided
            data = np.load(file)  # Load the scan as a numpy array
            if self.transform:
                data = self.transform(data)  # Apply transformations
            numpy_data.append(data)
            file_names.append(file)

        # Stack the data for the sequence of scans
        # Shape will be (num_scans, channels, D, H, W)
        numpy_data = np.stack(numpy_data, axis=0)

        # Return the data as a dictionary
        return {"file_names": file_names, "numpy": torch.tensor(numpy_data, dtype=torch.float32), "patient_id": patient_id}
