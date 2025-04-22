import itertools
from torch.utils.data import Dataset
import torch
from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
import random
import os

def split_data(ids, train_size=0.7, test_size=0.15, validation_size=0.15, seed=100):
    random.seed(seed)
    random.shuffle(ids)
    total_size = len(ids)
    train_end = int(train_size * total_size)
    test_end = train_end + int(test_size * total_size)
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
            scan_dates = sorted(os.listdir(os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for scan_date in scan_dates:
                numpy_file = os.path.join(self.numpy_dir, patient_id, scan_date)
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)
                    if 'PREBL00' in scan_date:
                        patient_label.append(0)
                    elif 'PREFU12' in scan_date:
                        patient_label.append(1)
                    else:
                        patient_label.append(2)
            all_data.append(patient_data)
            labels.append(patient_label)
        return all_data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        files = self.data[idx]
        label = self.labels[idx]

        if self.random_order:
            combined = list(zip(files, label))
            random.shuffle(combined)
            files, label = zip(*combined)

        numpy_data = [self.transform(np.load(file)) if self.transform else np.load(file) for file in files]
        numpy_data = np.stack(numpy_data, axis=0)
        return {"labels": torch.tensor(label, dtype=torch.long), "numpy": torch.tensor(numpy_data, dtype=torch.float32)}


class MRISliceDataLoader(Dataset):
    def __init__(self, numpy_dir, id_list, num_timepoints=3, transform=None, random_order=True):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.num_timepoints = num_timepoints
        self.random_order = random_order
        self.middle_slice_indices = list(range(44, 132))  # 88 central slices
        self.permutations = list(itertools.permutations(range(self.num_timepoints)))  # 6 possible permutations
        self.examples = self._prepare_examples(id_list)

    def _prepare_examples(self, id_list):
        examples = []  # Each = (ordered scan_slices [scan1, scan2, scan3], slice_index)
        for patient_id in id_list:
            patient_dir = os.path.join(self.numpy_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            scan_names = sorted(os.listdir(patient_dir))  # keep temporal order
            scan_paths = []
            for scan_name in scan_names:
                scan_dir = os.path.join(patient_dir, scan_name)
                
                if os.path.isdir(scan_dir):
                    slices = sorted([f for f in os.listdir(scan_dir) if f.endswith(".npy")])
                    
                    if len(slices) == 176:
                        scan_paths.append([os.path.join(scan_dir, f) for f in slices])

            if len(scan_paths) == self.num_timepoints:
                for slice_idx in self.middle_slice_indices:
                    examples.append((scan_paths, slice_idx))  # one example = 1 slice from 3 scans
        # print(len(examples))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        scan_paths, slice_idx = self.examples[idx]  # scan_paths = [scan1_slices, scan2_slices, scan3_slices]

        # Load the slice from each scan at the same index
        ordered_slices = []
        for scan in scan_paths:
            slice_path  = scan[slice_idx]
            img = np.load(slice_path)
            img = img.astype(np.float32)

            # Force it to be a 2D image for the transform
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # shape (H, W)
                else:
                    img = img[img.shape[0] // 2]  # use center slice

            if self.transform:
                img = self.transform(img)  # transform expects (H, W)
            ordered_slices.append(img)

        # Create the correct label (original order is [0, 1, 2])
        original_order = list(range(self.num_timepoints))

        # Optionally permute the order
        if self.random_order:
            combined = list(zip(ordered_slices, original_order))
            random.shuffle(combined)
            ordered_slices, permuted_order = zip(*combined)
        else:
            permuted_order = original_order

        # Convert permuted_order into class label (0 to 5)
        label = self.permutations.index(tuple(permuted_order))
        data = np.stack(ordered_slices, axis=0)  # shape: [3, H, W]
        return {
            "numpy": torch.tensor(data, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "slice_index": torch.tensor(slice_idx, dtype=torch.long)
        }
class MRIGenerationLoader(Dataset):
    def __init__(self, root_dir, id_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.middle_slice_indices = list(range(44, 132))  # 88 slices
        self.examples = self._prepare_data(id_list)

    def _prepare_data(self, id_list):
        examples = []
        for patient_id in id_list:
            patient_path = os.path.join(self.root_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue

            scan_names = sorted(os.listdir(patient_path))  # e.g., scan1, scan2, ..., scan5
            scan_paths = []
            for scan_name in scan_names:
                scan_dir = os.path.join(patient_path, scan_name)
                if os.path.isdir(scan_dir):
                    slice_files = sorted([f for f in os.listdir(scan_dir) if f.endswith(".npy")])
                    if len(slice_files) == 176:
                        scan_paths.append([os.path.join(scan_dir, f) for f in slice_files])

            if len(scan_paths) >= 5:
                for slice_idx in self.middle_slice_indices:
                    examples.append((scan_paths[:5], slice_idx))  # Use first 5 scans
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        scan_paths, slice_idx = self.examples[idx]  # scan_paths = list of 5 scan_slices

        # Load all 5 slices at the same index
        slices = []
        for scan in scan_paths:
            slice_path  = scan[slice_idx]
            img = np.load(slice_path)
            img = img.astype(np.float32)

            # Force it to be a 2D image for the transform
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # shape (H, W)
                else:
                    img = img[img.shape[0] // 2]  # use center slice

            if self.transform:
                img = self.transform(img)  # transform expects (H, W)
            slices.append(img)

        slices = np.stack(slices, axis=0)  # Shape: [5, H, W]

        input_seq = slices[:4]  # First 4 as input
        target_slice = slices[4]  # 5th as target

        return {
            "input": torch.tensor(input_seq).unsqueeze(1).float(),   # Shape: [4, 1, H, W]
            "target": torch.tensor(target_slice).unsqueeze(0).float()  # Shape: [1, H, W]
        }
