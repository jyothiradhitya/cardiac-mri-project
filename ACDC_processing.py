# acdc_processing_fixed.py
import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

# Root ACDC dataset
input_dirs = ['./Database/training', './Database/testing']
RESIZE_SHAPE = (128, 128)

# Normalize image
def normalize_image(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)

# Scale mask values 0–3 to 0–255 for visibility
def scale_mask(mask):
    return (mask * 85).astype(np.uint8)

for root in input_dirs:
    split = 'train' if 'training' in root else 'test'
    out_img_dir = f'./dataset/{split}/images'
    out_mask_dir = f'./dataset/{split}/masks'
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    print(f"\n📁 Processing {split.upper()} from: {root}")

    for patient in tqdm(os.listdir(root), desc=f"🔍 Patients in {split}"):
        patient_path = os.path.join(root, patient)
        if not os.path.isdir(patient_path): continue

        nii_files = [f for f in os.listdir(patient_path) if f.endswith('.nii') and '_gt' not in f]

        for nii_file in nii_files:
            image_path = os.path.join(patient_path, nii_file)
            mask_path = image_path.replace('.nii', '_gt.nii')

            if not os.path.exists(mask_path):
                print(f"⚠️ Missing mask for: {nii_file}")
                continue

            image_vol = nib.load(image_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()

            for i in range(image_vol.shape[2]):
                img_slice = normalize_image(image_vol[:, :, i])
                mask_slice = mask_vol[:, :, i].astype(np.uint8)

                img_resized = cv2.resize(img_slice, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_slice, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)

                base_name = f"{nii_file.replace('.nii','')}_slice{i:02d}"
                img_file = os.path.join(out_img_dir, base_name + ".png")
                mask_file = os.path.join(out_mask_dir, base_name + "_mask.png")

                # Only save valid masks
                if mask_resized.sum() == 0:
                    continue

                cv2.imwrite(img_file, (img_resized * 255).astype(np.uint8))
                cv2.imwrite(mask_file, scale_mask(mask_resized))

            print(f"✅ Processed: {nii_file} with {image_vol.shape[2]} slices")
