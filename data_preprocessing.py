import nibabel as nib
import numpy as np
import os
import glob
from tqdm import tqdm

# Define input and output directories
input_dir = "data_t1w_3_scans"  # Change this to your MINC dataset directory
output_dir = "numpy_conversions"  # Output directory for NumPy files
os.makedirs(output_dir, exist_ok=True)

# List all MINC files
minc_files = glob.glob(os.path.join(input_dir, "**", "**", "*.mnc"), recursive=True)

# Convert each MINC file to NumPy format
for file in tqdm(minc_files, desc="Converting MINC to NumPy"):
    # Load the MINC file
    img = nib.load(file)
    
    # Get image data as a NumPy array
    img_data = img.get_fdata()

    # Define the relative path for the output folder (keeping the structure)
    relative_path = os.path.relpath(file, input_dir)
    output_file = os.path.join(output_dir, relative_path.replace(".mnc", ".npy"))

    # Ensure the target folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save as NumPy file
    np.save(output_file, img_data)

print("Conversion complete! NumPy files are saved in:", output_dir)