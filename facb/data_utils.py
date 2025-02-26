import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

IMAGE_SIZE = (224, 224)

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.7),  
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),  
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),  
    A.OneOf([
        A.RandomShadow(p=0.3),
        A.RandomFog(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ], p=0.5),
    A.OneOf([
        A.ISONoise(p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
    ], p=0.5),
    A.Normalize(mean=[0.5], std=[0.5]),  
    ToTensorV2(),
])

resize_transform = transforms.Resize(IMAGE_SIZE)

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def fft_transform(self, img_tensor):
        img_tensor = img_tensor[:3, :, :]
        img_np = img_tensor.cpu().numpy()  # Ensure conversion happens on CPU
        fft_img = np.fft.fft2(img_np, axes=(-2, -1))
        magnitude = np.abs(np.fft.fftshift(fft_img, axes=(-2, -1)))
        magnitude = np.log1p(magnitude)
        return torch.tensor(magnitude, dtype=torch.float32)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(resize_transform(image))  # Convert to NumPy array for Albumentations
        augmented = augmentations(image=image)
        x = augmented["image"]
        y = augmentations(image=image)["image"]
        return x, self.fft_transform(x), y, self.fft_transform(y)


def get_dataloader(root_dir, batch_size=8, num_workers=4):  # Increased num_workers
    dataset = CustomDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)  # Added pin_memory for better GPU efficiency