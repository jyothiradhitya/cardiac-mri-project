# acdc_healthy_unhealthy_fixed.py
import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

RESIZE_SHAPE = (128, 128)

def normalize_image(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)

def scale_mask(mask):
    return (mask * 85).astype(np.uint8)

def categorize_patients(acdc_path):
    healthy, unhealthy = [], []
    for patient in os.listdir(acdc_path):
        cfg_path = os.path.join(acdc_path, patient, "Info.cfg")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                for line in f:
                    if line.startswith("Group"):
                        if "NOR" in line:
                            healthy.append(patient)
                        else:
                            unhealthy.append(patient)
    return healthy, unhealthy

def convert_and_save(patient_path, patient_id, out_img_dir, out_mask_dir):
    image_path = os.path.join(patient_path, f"{patient_id}_frame01.nii")
    mask_path = os.path.join(patient_path, f"{patient_id}_frame01_gt.nii")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"❌ Missing image or mask for {patient_id}")
        return

    img_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    for i in range(img_data.shape[2]):
        img_slice = normalize_image(img_data[:, :, i])
        mask_slice = mask_data[:, :, i].astype(np.uint8)

        img_resized = cv2.resize(img_slice, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, RESIZE_SHAPE, interpolation=cv2.INTER_NEAREST)

        if mask_resized.sum() == 0:
            continue  # skip empty masks

        base_name = f"{patient_id}_slice{i:02d}"
        img_file = os.path.join(out_img_dir, base_name + ".png")
        mask_file = os.path.join(out_mask_dir, base_name + "_mask.png")

        cv2.imwrite(img_file, (img_resized * 255).astype(np.uint8))
        cv2.imwrite(mask_file, scale_mask(mask_resized))

def process_category(acdc_path, patient_list, category):
    img_dir = os.path.join(category, "images")
    mask_dir = os.path.join(category, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for patient in tqdm(patient_list, desc=f"Processing {category}"):
        patient_path = os.path.join(acdc_path, patient)
        convert_and_save(patient_path, patient, img_dir, mask_dir)

# ==== MAIN ====
input_acdc_path = "./Database/training"
healthy_patients, unhealthy_patients = categorize_patients(input_acdc_path)

print(f"✅ Healthy patients: {len(healthy_patients)}")
print(f"❗ Unhealthy patients: {len(unhealthy_patients)}")

process_category(input_acdc_path, healthy_patients, "healthy")
process_category(input_acdc_path, unhealthy_patients, "unhealthy")
