# Train Test Val Split

# 80 - 10 - 10 split

import itertools
from torch.utils.data import Dataset
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


class MRISliceDataLoader(Dataset):
    def __init__(self, numpy_dir, id_list, random_order=True, transform=None):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.permutations = list(itertools.permutations([0, 1, 2]))
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        all_labels = []
        for patient_id in self.patient_ids:
            scan_dates = sorted(os.listdir(
                os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            scan_order = []
            for scan_date in scan_dates:
                numpy_file = os.path.join(
                    self.numpy_dir, patient_id, scan_date,
                    f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy"
                )
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)
                    if scan_date == 'PREBL00':
                        scan_order.append(0)
                    elif scan_date == 'PREFU12':
                        scan_order.append(1)
                    else:
                        scan_order.append(2)

            # Only keep samples with exactly 3 timepoints
            if len(patient_data) == 3:
                all_data.append(patient_data)
                all_labels.append(scan_order)
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numpy_files = self.data[idx]
        true_order = self.labels[idx]

        # Zip and shuffle together
        combined = list(zip(numpy_files, true_order))
        if self.random_order:
            random.shuffle(combined)
        shuffled_files, shuffled_order = zip(*combined)

        # Convert the new permutation to a class label (0-5)
        permutation_label = self.permutations.index(tuple(shuffled_order))

        # Load data
        numpy_data = []
        for file in shuffled_files:
            data = np.load(file)
            if self.transform:
                data = self.transform(data)
            numpy_data.append(data)

        # Stack into (3, D, H, W)
        numpy_data = np.stack(numpy_data, axis=0)

        # Randomly choose a slice along depth D
        d_idx = np.random.randint(numpy_data.shape[1])
        slices_2d = numpy_data[:, d_idx, :, :]  # Shape: (3, H, W)

        return {
            "numpy": torch.tensor(slices_2d, dtype=torch.float32),
            "label": torch.tensor(permutation_label, dtype=torch.long)
        }


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


class MRIGenerationLoader(Dataset):
    def __init__(self, root_dir, id_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.id_list = id_list
        self.data = self._prepare_data()

    def _prepare_data(self):
        valid_samples = []
        for patient_id in self.id_list:
            patient_path = os.path.join(self.root_dir, patient_id)
            if not os.path.exists(patient_path):
                continue

            timepoint_order = ['PREBL00', 'PREFU12',
                               'PREFU24', 'PREFU36', 'PREFU48']
            scan_dates = sorted(os.listdir(patient_path),
                                key=lambda x: timepoint_order.index(x))

            patient_scans = []

            for date in scan_dates:
                scan_dir = os.path.join(patient_path, date)
                npy_files = [f for f in os.listdir(
                    scan_dir) if f.endswith('.npy')]
                if not npy_files:
                    continue
                full_path = os.path.join(scan_dir, npy_files[0])
                patient_scans.append(full_path)

            if len(patient_scans) >= 5:
                valid_samples.append(patient_scans[:5])  # T0–T4

        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]

        volumes = [np.load(p) for p in paths]  # Each: [D, H, W]
        if self.transform:
            volumes = [self.transform(vol) for vol in volumes]

        volumes = np.stack(volumes, axis=0)  # [5, D, H, W]
        d_idx = np.random.randint(volumes.shape[1])
        slice_2d = volumes[:, d_idx, :, :]  # [5, H, W]

        input_seq = slice_2d[:4]      # [4, H, W]
        target_slice = slice_2d[4]    # [H, W]

        return {
            # [4, 1, H, W]
            "input": torch.tensor(input_seq).unsqueeze(1).float(),
            # [1, H, W]
            "target": torch.tensor(target_slice).unsqueeze(0).float()
        }
