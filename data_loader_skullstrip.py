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
    def __init__(self, numpy_dir, id_list, num_timepoints=3, random_order=True, transform=None):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.num_timepoints = num_timepoints
        self.permutations = list(itertools.permutations(range(self.num_timepoints)))
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        all_labels = []
        for patient_id in self.patient_ids:
            scan_dir = os.path.join(self.numpy_dir, patient_id)
            if not os.path.isdir(scan_dir):
                continue
            scan_dates = sorted(os.listdir(scan_dir))
            patient_data = []
            scan_order = []

            for i, scan_file in enumerate(scan_dates):
                numpy_file = os.path.join(scan_dir, scan_file)
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)
                    scan_order.append(i)

            if len(patient_data) == self.num_timepoints:
                all_data.append(patient_data)
                all_labels.append(scan_order)
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numpy_files = self.data[idx]
        true_order = self.labels[idx]

        combined = list(zip(numpy_files, true_order))
        if self.random_order:
            random.shuffle(combined)
        shuffled_files, shuffled_order = zip(*combined)
        permutation_label = self.permutations.index(tuple(shuffled_order))

        numpy_data = [self.transform(np.load(file)) if self.transform else np.load(file) for file in shuffled_files]
        numpy_data = np.stack(numpy_data, axis=0)
        d_idx = np.random.randint(numpy_data.shape[1])
        slices_2d = numpy_data[:, d_idx, :, :]

        return {
            "numpy": torch.tensor(slices_2d, dtype=torch.float32),
            "label": torch.tensor(permutation_label, dtype=torch.long)
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
            scan_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.npy')])
            patient_scans = []
            for scan_file in scan_files:
                full_path = os.path.join(patient_path, scan_file)
                patient_scans.append(full_path)
            if len(patient_scans) >= 5:
                valid_samples.append(patient_scans[:5])
        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]
        volumes = [self.transform(np.load(file)) if self.transform else np.load(file) for file in paths]
        volumes = np.stack(volumes, axis=0)
        d_idx = np.random.randint(volumes.shape[1])
        slice_2d = volumes[:, d_idx, :, :]
        input_seq = slice_2d[:4]
        target_slice = slice_2d[4]
        return {
            "input": torch.tensor(input_seq).unsqueeze(1).float(),
            "target": torch.tensor(target_slice).unsqueeze(0).float()
        }
