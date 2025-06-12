import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from mri_dataset import MRIDataset  # ✅ import your dataset loader

# ✅ UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.down1 = conv_block(1, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up2 = conv_block(256 + 128, 128)
        self.up1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        u2 = self.up(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c4 = self.up2(u2)
        u1 = self.up(c4)
        u1 = torch.cat([u1, c1], dim=1)
        c5 = self.up1(u1)
        out = self.final(c5)
        return torch.sigmoid(out)

# ✅ Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# ✅ Training Function
def train_model():
    print("🚀 Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ✅ Load datasets
    train_dataset = MRIDataset('./dataset/train/images', './dataset/train/masks')
    test_dataset = MRIDataset('./dataset/test/images', './dataset/test/masks')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # ✅ Initialize model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'unet_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), 'unet_model_final.pth')
    print("✅ Training complete, model saved.")
    return model

# ✅ Start training
if __name__ == "__main__":
    try:
        trained_model = train_model()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
