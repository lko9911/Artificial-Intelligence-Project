# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DepthDataset(Dataset):
    def __init__(self, root_dir, indices=None, img_size=(512, 256)):
        self.image_dir = os.path.join(root_dir, "images")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.depth_files = sorted(os.listdir(self.depth_dir))
        self.img_size = img_size

        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- Image ---
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size)
        image = transforms.ToTensor()(image)  # (C,H,W), 0~1

        # --- Depth ---
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32") / 1000.0
        depth[depth <= 0] = 0.01
        depth = cv2.resize(depth, self.img_size)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1,H,W)

        return image, depth
