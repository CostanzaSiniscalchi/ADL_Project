"""
Filename: strip_all_skulls.py
Author: Costanza Siniscalchi
Description: Batch skull-stripping pipeline for converting Numpy MRI volumes to NIfTI,
             applying FSL's BET, and saving stripped output.
"""


import os
import numpy as np
import nibabel as nib
import tempfile
import subprocess

def skull_strip_and_save_all(numpy_dir, cache_dir="stripped_cache"):
    # Create directory to cache skull-stripped volumes
    os.makedirs(cache_dir, exist_ok=True)
    patient_ids = sorted(os.listdir(numpy_dir))

    for patient_id in patient_ids:
        patient_path = os.path.join(numpy_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        scan_dates = sorted(os.listdir(patient_path))
        for scan_date in scan_dates:
            # Define input .npy file path
            npy_name = f"preventad_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy"
            npy_path = os.path.join(patient_path, scan_date, npy_name)

            if not os.path.isfile(npy_path):
                print(f"Missing: {npy_path}")
                continue

            # Define output path for the skull-stripped .npy
            output_path = os.path.join(
                cache_dir, patient_id,
                f"preventad_stripped_{patient_id}_{scan_date}_t1w_001_t1w-defaced_001.npy"
            )
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"Already exists: {output_path}")
                continue

            # Ensure patient-specific subdirectory exists in cache
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load original volume
            volume = np.load(npy_path)

            # Use a temporary directory for intermediate NIfTI files
            with tempfile.TemporaryDirectory() as tmpdir:
                nii_path = os.path.join(tmpdir, "input.nii.gz")
                out_path = os.path.join(tmpdir, "output.nii.gz")

                # Save .npy volume as a NIfTI file to run FSL's BET
                nib.save(nib.Nifti1Image(volume, np.eye(4)), nii_path)

                try:
                    # Run BET (Brain Extraction Tool)
                    subprocess.run(["bet", nii_path, out_path, "-f", "0.5", "-g", "0", "-m"], check=True)
                    
                    # Load stripped output and save as .npy
                    stripped = nib.load(out_path).get_fdata().astype(np.float32)
                    np.save(output_path, stripped)
                    print(f"Stripped and saved: {output_path}")
                except subprocess.CalledProcessError:
                    print(f"BET failed for {patient_id} {scan_date}. Skipping.")

# Example usage
skull_strip_and_save_all(
    "/Users/costanzasiniscalchi/Documents/MS/ADL/Project/ADL_Project/data/numpy_conversions_5_scans",
    "/Users/costanzasiniscalchi/Documents/MS/ADL/Project/ADL_Project/data/stripped_5_scans"
)
