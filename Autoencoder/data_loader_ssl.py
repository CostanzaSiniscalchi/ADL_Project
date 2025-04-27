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


class MRIDataLoader(Dataset):
    def __init__(self, numpy_dir, id_list, transform=None, mask_scan=None, mask_ratio=0.75):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.mask_scan = mask_scan
        self.mask_ratio = mask_ratio
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        labels = []
        for patient_id in self.patient_ids:
            filenames = sorted(os.listdir(
                os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for filename in filenames:
                file_path = os.path.join(self.numpy_dir, patient_id, filename)
                if os.path.exists(file_path):
                    patient_data.append(file_path)
                    if 'PREBL00' in filename:
                        patient_label.append(0)
                    elif 'PREFU12' in filename:
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
        label = self.labels[idx]
        numpy_data = []

        for file in numpy_files:
            data = np.load(file)
            # use the middle 32 slices
            data = data[data.shape[0]//2 - 16: data.shape[0]//2 + 16]
            if self.transform:
                data = self.transform(data)
            numpy_data.append(data)

        # Convert to stacked 3-channel input
        numpy_data = np.stack(numpy_data, axis=0)
        # need to clip??
        numpy_data = np.clip(numpy_data, 0.0, 1.0)

        # Randomly mask one scan (for SSL)
        original_data = numpy_data.copy()
        masked_index = -1
        if self.mask_scan == 'total':
            masked_index = np.random.randint(0, len(numpy_data))
            original_shape = numpy_data[masked_index].shape
            numpy_data[masked_index] = np.zeros(
                original_shape, dtype=np.float32)
        elif self.mask_scan == 'random':
            # Masking ratio
            masked_index = np.random.randint(0, len(numpy_data))
            # Create random binary mask with 40% of voxels zeroed out
            if self.mask_ratio == 'random':
                ratio = np.random.uniform(0.25, 0.5)
            else:
                ratio = self.mask_ratio
            mask = np.random.rand(
                *numpy_data[masked_index].shape) > ratio

            # Apply elementwise mask
            numpy_data[masked_index] = numpy_data[masked_index] * \
                mask.astype(np.float32)

        return {
            "labels": torch.tensor(label, dtype=torch.long),
            "numpy": torch.tensor(numpy_data, dtype=torch.float32),  # masked
            "original": torch.tensor(original_data, dtype=torch.float32),
            "masked_index": masked_index
        }


class MRISliceDataLoader(Dataset):
    def __init__(self, numpy_dir, id_list, num_timepoints=3, transform=None, mask_scan=None, mask_ratio=0.75):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.num_timepoints = num_timepoints
        self.mask_scan = mask_scan
        self.mask_ratio = mask_ratio
        self.middle_slice_indices = list(range(44, 132))  # 88 central slices
        self.examples = self._prepare_examples(id_list)

    def _prepare_examples(self, id_list):
        examples = []
        for patient_id in id_list:
            patient_dir = os.path.join(self.numpy_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            scan_names = sorted(os.listdir(patient_dir))  # keep temporal order
            scan_paths = []
            for scan_name in scan_names:
                scan_dir = os.path.join(patient_dir, scan_name)
                if os.path.isdir(scan_dir):
                    slices = sorted([f for f in os.listdir(
                        scan_dir) if f.endswith(".npy")])
                    if len(slices) == 176:
                        scan_paths.append(
                            [os.path.join(scan_dir, f) for f in slices])

            if len(scan_paths) == self.num_timepoints:
                for slice_idx in self.middle_slice_indices:
                    examples.append((scan_paths, slice_idx))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        scan_paths, slice_idx = self.examples[idx]

        ordered_slices = []
        original_slices = []

        for scan in scan_paths:
            slice_path = scan[slice_idx]
            img = np.load(slice_path).astype(np.float32)  # shape: (H, W)

            if self.transform:
                img = self.transform(img)

            original_slices.append(img)

        # Convert list to tensor: [3, H, W]
        original_data = np.stack(original_slices, axis=0)

        # need to clip??
        original_data = np.clip(original_data, 0.0, 1.0)

        # Apply masking
        masked_data = original_data.copy()
        masked_index = -1

        if self.mask_scan == 'total':
            masked_index = np.random.randint(0, self.num_timepoints)
            # or torch.zeros_like(masked_data[masked_index])
            masked_data[masked_index] = 0.0
        elif self.mask_scan == 'random':
            # Masking ratio
            masked_index = np.random.randint(0, self.num_timepoints)

            # Create random binary mask with 40% of voxels zeroed out
            if self.mask_ratio == 'random':
                ratio = np.random.uniform(0.25, 0.75)
            else:
                ratio = self.mask_ratio
            mask = np.random.rand(
                *masked_data[masked_index].shape) > ratio

            # Apply elementwise mask
            masked_data[masked_index] = masked_data[masked_index] * \
                mask.astype(np.float32)

        return {
            # masked
            "numpy": torch.tensor(masked_data, dtype=torch.float32),
            # unmasked for supervision
            "original": torch.tensor(original_data, dtype=torch.float32),
            "masked_index": masked_index,
            "slice_index": torch.tensor(slice_idx, dtype=torch.long)
        }


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
            order_map = {k: i for i, k in enumerate(timepoint_order)}

            scan_dates = sorted(os.listdir(patient_path),
                                key=lambda x: next((order_map[k] for k in timepoint_order if k in x), float('inf')))

            patient_scans = []

            for npy_file in scan_dates:
                full_path = os.path.join(patient_path, npy_file)
                patient_scans.append(full_path)

            if len(patient_scans) >= 5:
                valid_samples.append(patient_scans[:5])  # T0–T4

        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]

        volumes = [np.load(p) for p in paths]  # Each: [D, H, W]
        for v in range(len(volumes)):
            # use the middle 32 slices
            volumes[v] = volumes[v][volumes[v].shape[0] //
                                    2 - 16: volumes[v].shape[0]//2 + 16]
            if self.transform:
                volumes[v] = self.transform(volumes[v])

        # Convert to stacked 3-channel input
        numpy_data = np.stack(volumes, axis=0)
        # need to clip??
        numpy_data = np.clip(numpy_data, 0.0, 1.0)

        input_seq = numpy_data[:4]
        target_vol = numpy_data[4:5]

        return {
            "input": torch.tensor(input_seq, dtype=torch.float32),
            "target": torch.tensor(target_vol, dtype=torch.float32),
        }
