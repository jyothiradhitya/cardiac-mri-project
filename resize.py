# fix_resize_and_save.py
import os
import cv2
from tqdm import tqdm

def resize_and_save_all(image_dir, mask_dir, out_image_dir, out_mask_dir, size=(128, 128)):
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for fname in tqdm(os.listdir(image_dir), desc=f"Resizing {image_dir}"):
        if not fname.endswith(".png"): continue

        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"❌ Skipped (not found): {fname}")
            continue

        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(out_image_dir, fname), img_resized)
        cv2.imwrite(os.path.join(out_mask_dir, fname.replace(".png", "_mask.png")), mask_resized)

    print(f"✅ Done resizing {image_dir}")

# === Run for Healthy & Unhealthy ===
resize_and_save_all(
    image_dir="./healthy/images",
    mask_dir="./healthy/masks",
    out_image_dir="./healthy_resized/images",
    out_mask_dir="./healthy_resized/masks"
)

resize_and_save_all(
    image_dir="./unhealthy/images",
    mask_dir="./unhealthy/masks",
    out_image_dir="./unhealthy_resized/images",
    out_mask_dir="./unhealthy_resized/masks"
)
