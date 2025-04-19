import itertools
from torch.utils.data import Dataset
from math import comb
import torch
from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
import random
import os
import nibabel as nib
import tempfile
import subprocess
import itertools


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


def skull_strip_with_bet(volume, patient_id=None, scan_date=None, cache_dir="stripped_cache"):
    if patient_id and scan_date:
        cache_path = os.path.join(cache_dir, patient_id, f"{scan_date}.npy")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        nii_path = os.path.join(tmpdir, "input.nii.gz")
        out_path = os.path.join(tmpdir, "output.nii.gz")
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume, affine), nii_path)
        try:
            subprocess.run(["bet", nii_path, out_path, "-f", "0.5", "-g", "0", "-m"], check=True)
            stripped = nib.load(out_path).get_fdata().astype(np.float32)
            if patient_id and scan_date:
                np.save(cache_path, stripped)
            return stripped
        except subprocess.CalledProcessError:
            print(f"BET failed for {patient_id} {scan_date}, using unstripped volume.")
            return volume


class MRIDataLoader:
    def __init__(self, numpy_dir, id_list, random_order=True, transform=None, cache_dir="stripped_cache"):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.cache_dir = cache_dir
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        labels = []
        for patient_id in self.patient_ids:
            scan_dates = sorted(os.listdir(os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for scan_date in scan_dates:
                numpy_file = os.path.join(self.numpy_dir, patient_id, scan_date,
                                          f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy")
                if os.path.exists(numpy_file):
                    patient_data.append((numpy_file, scan_date))
                    if scan_date == 'PREBL00':
                        patient_label.append(0)
                    elif scan_date == 'PREFU12':
                        patient_label.append(1)
                    else:
                        patient_label.append(2)
            all_data.append((patient_id, patient_data))
            labels.append(patient_label)
        return all_data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_id, files_and_dates = self.data[idx]
        label = self.labels[idx]

        if self.random_order:
            combined = list(zip(files_and_dates, label))
            random.shuffle(combined)
            files_and_dates, label = zip(*combined)

        numpy_data = []
        for file, scan_date in files_and_dates:
            data = np.load(file)
            data = skull_strip_with_bet(data, patient_id, scan_date, self.cache_dir)
            if self.transform:
                data = self.transform(data)
            numpy_data.append(data)

        numpy_data = np.stack(numpy_data, axis=0)
        return {"labels": torch.tensor(label, dtype=torch.long), "numpy": torch.tensor(numpy_data, dtype=torch.float32)}


class MRISliceDataLoader(Dataset):
    def __init__(self, numpy_dir, id_list, num_timepoints=3, random_order=True, transform=None, cache_dir="stripped_cache"):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.cache_dir = cache_dir
        self.num_timepoints = num_timepoints
        os.makedirs(self.cache_dir, exist_ok=True)
        self.permutations = list(itertools.permutations(range(self.num_timepoints)))
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        all_labels = []
        for patient_id in self.patient_ids:
            scan_dir = os.path.join(self.numpy_dir, patient_id)
            if not os.path.isdir(scan_dir):
                print(f"missed {scan_dir}")
                continue
            scan_dates = sorted(os.listdir(scan_dir))
            patient_data = []
            scan_order = []

            for i, scan_date in enumerate(scan_dates):
                numpy_file = os.path.join(
                    self.numpy_dir, patient_id, scan_date,
                    f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy"
                )
                if os.path.exists(numpy_file):
                    patient_data.append(numpy_file)
                    scan_order.append(i)  # use index as order placeholder

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

        numpy_data = []
        for file in shuffled_files:
            data = np.load(file)

            parts = file.split(os.sep)
            patient_id = parts[-3]
            scan_date = parts[-2]

            stripped = self.skull_strip_with_bet(data, patient_id, scan_date)
            

            if self.transform:
                stripped = self.transform(stripped)

            numpy_data.append(stripped)

        numpy_data = np.stack(numpy_data, axis=0)
        d_idx = np.random.randint(numpy_data.shape[1])
        slices_2d = numpy_data[:, d_idx, :, :]

        return {
            "numpy": torch.tensor(slices_2d, dtype=torch.float32),
            "label": torch.tensor(permutation_label, dtype=torch.long)
        }

    def skull_strip_with_bet(self, volume, patient_id=None, scan_date=None):
        if patient_id and scan_date:
            cache_path = os.path.join(self.cache_dir, patient_id, f"{scan_date}.npy")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                return np.load(cache_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = os.path.join(tmpdir, "input.nii.gz")
            out_path = os.path.join(tmpdir, "output.nii.gz")

            affine = np.eye(4)
            nib.save(nib.Nifti1Image(volume, affine), nii_path)

            try:
                subprocess.run(
                    ["bet", nii_path, out_path, "-f", "0.5", "-g", "0", "-m"],
                    check=True
                )
                stripped = nib.load(out_path).get_fdata().astype(np.float32)

                if patient_id and scan_date:
                    np.save(cache_path, stripped)
                
                print("stripped image")

                return stripped
            except subprocess.CalledProcessError:
                print(f"BET failed for {patient_id} {scan_date}, using unstripped volume.")
                return volume
            
    def show_stripped_example(self, pid, save_dir = 'skull_stripped_examples'):
        scan_dates = os.listdir(os.path.join( self.cache_dir, pid))
        for date in scan_dates:
            npy_path = os.path.join( self.cache_dir, pid, date)
            if not os.path.isfile(npy_path):
                continue
            stripped_volume = np.load(npy_path)  # [D, H, W]

            mid_slice = stripped_volume[stripped_volume.shape[0] // 2]

            plt.imshow(mid_slice, cmap='gray')
            plt.title(f"{pid} - {date}")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"skull_strippped_{pid}_{date}.png"))
            plt.close()

class EmbeddedMRIDataLoader:
    def __init__(self, numpy_dir, id_list, random_order=True, transform=None, cache_dir="stripped_cache"):
        self.numpy_dir = numpy_dir
        self.transform = transform
        self.patient_ids = id_list
        self.random_order = random_order
        self.cache_dir = cache_dir
        self.data, self.labels = self._prepare_data()

    def _prepare_data(self):
        all_data = []
        labels = []
        for patient_id in self.patient_ids:
            numpy_files = sorted(os.listdir(os.path.join(self.numpy_dir, patient_id)))
            patient_data = []
            patient_label = []
            for numpy_file in numpy_files:
                patient_data.append((numpy_file, patient_id))
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
        files_and_ids = self.data[idx]
        label = self.labels[idx]

        if self.random_order:
            combined = list(zip(files_and_ids, label))
            random.shuffle(combined)
            files_and_ids, label = zip(*combined)

        numpy_data = []
        for file_name, patient_id in files_and_ids:
            file_path = os.path.join(self.numpy_dir, patient_id, file_name)
            scan_date = file_name.split("_")[2]
            data = np.load(file_path)
            data = skull_strip_with_bet(data, patient_id, scan_date, self.cache_dir)
            if self.transform:
                data = self.transform(data)
            numpy_data.append(data)

        numpy_data = np.stack(numpy_data, axis=0)
        return {"labels": torch.tensor(label, dtype=torch.long), "numpy": torch.tensor(numpy_data, dtype=torch.float32)}


class MRIGenerationLoader(Dataset):
    def __init__(self, root_dir, id_list, transform=None, cache_dir="stripped_cache"):
        self.root_dir = root_dir
        self.transform = transform
        self.id_list = id_list
        self.cache_dir = cache_dir
        self.data = self._prepare_data()

    def _prepare_data(self):
        valid_samples = []
        for patient_id in self.id_list:
            patient_path = os.path.join(self.root_dir, patient_id)
            if not os.path.exists(patient_path):
                continue
            timepoint_order = ['PREBL00', 'PREFU12', 'PREFU24', 'PREFU36', 'PREFU48']
            scan_dates = sorted(os.listdir(patient_path), key=lambda x: timepoint_order.index(x))
            patient_scans = []
            for date in scan_dates:
                scan_dir = os.path.join(patient_path, date)
                npy_files = [f for f in os.listdir(scan_dir) if f.endswith('.npy')]
                if not npy_files:
                    continue
                full_path = os.path.join(scan_dir, npy_files[0])
                patient_scans.append((full_path, patient_id, date))
            if len(patient_scans) >= 5:
                valid_samples.append(patient_scans[:5])
        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]
        volumes = []
        for file_path, patient_id, scan_date in paths:
            vol = np.load(file_path)
            vol = skull_strip_with_bet(vol, patient_id, scan_date, self.cache_dir)
            if self.transform:
                vol = self.transform(vol)
            volumes.append(vol)
        volumes = np.stack(volumes, axis=0)
        d_idx = np.random.randint(volumes.shape[1])
        slice_2d = volumes[:, d_idx, :, :]
        input_seq = slice_2d[:4]
        target_slice = slice_2d[4]
        return {
            "input": torch.tensor(input_seq).unsqueeze(1).float(),
            "target": torch.tensor(target_slice).unsqueeze(0).float()
        }
