# dataset/mri_dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_mask_pairs = []

        for fname in os.listdir(image_dir):
            if fname.endswith(".png"):
                image_path = os.path.join(image_dir, fname)
                mask_name = fname.replace(".png", "_mask.png")
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((image_path, mask_path))
                else:
                    print(f"⚠ Mask not found for {fname}, skipping...")

        print(f"✅ Loaded {len(self.image_mask_pairs)} image-mask pairs.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]

        # Load and normalize
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to load image or mask at index {idx}")

        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0)
        mask = (mask > 0.5).astype(np.float32)

        # Fix negative stride issue
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
