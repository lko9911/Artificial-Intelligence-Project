import os
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image


# ==========================
# Config 설정
# ==========================
class Config:
    root_dir = "Depth/Dataset"
    checkpoint_dir = "./checkpoints_depth"
    batch_size = 4
    learning_rate = 1e-4
    num_workers = 4
    epochs = 44
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 42


# ==========================
# 재현성 보장
# ==========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================
# DepthDataset
# ==========================
class DepthDataset(Dataset):
    def __init__(self, root_dir, indices=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.depth_files = sorted(os.listdir(self.depth_dir))

        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]

        self.img_size = (512, 256)  # width, height

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        image = transforms.ToTensor()(image)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32") / 1000.0
        depth[depth <= 0] = 0.01
        depth = cv2.resize(depth, self.img_size)
        depth = torch.from_numpy(depth).unsqueeze(0)

        return image, depth


# ==========================
# 모델 정의
# ==========================
class DeepLabv3Plus_Depth(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[-1] = nn.Sequential(
            nn.Conv2d(256, output_channels, kernel_size=1),
            nn.Softplus()  # 더 안정적인 양수 출력
        )

    def forward(self, x):
        out = self.model(x)["out"]
        return torch.nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ==========================
# Scale-Invariant Loss
# ==========================
def scale_invariant_loss(pred, target):
    if pred.shape != target.shape:
        target = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='nearest')

    valid_mask = (target > 0)
    pred = pred[valid_mask]
    target = target[valid_mask]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)

    diff = torch.log(pred) - torch.log(target)
    loss = torch.mean(diff ** 2) - (torch.mean(diff) ** 2)
    return loss


# ==========================
# Train / Validation
# ==========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", ncols=100)
    for imgs, depths in pbar:
        imgs, depths = imgs.to(device), depths.to(device)
        preds = model(imgs)
        loss = criterion(preds, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, depths in tqdm(loader, desc="Validation", ncols=100):
            imgs, depths = imgs.to(device), depths.to(device)
            preds = model(imgs)
            loss = criterion(preds, depths)
            total_loss += loss.item()
    return total_loss / len(loader)


# ==========================
# Checkpoint 함수
# ==========================
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss
    }, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["val_loss"]


# ==========================
# Dataset Split
# ==========================
def split_dataset(dataset, cfg):
    total_size = len(dataset)
    train_size = int(total_size * cfg.train_ratio)
    val_size = int(total_size * cfg.val_ratio)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    random.seed(cfg.seed)
    random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    return train_idx, val_idx, test_idx


# ==========================
# Visualization
# ==========================
def visualize_predictions(model, dataset, device, num_samples=5, save_dir="./results_depth"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    with torch.no_grad():
        for i in indices:
            image, depth_gt = dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            pred = model(image_batch).squeeze().cpu().numpy()

            # 시각화용 정규화
            pred_vis = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-6)
            gt_vis = depth_gt.squeeze().numpy()
            gt_vis = (gt_vis - np.min(gt_vis)) / (np.max(gt_vis) - np.min(gt_vis) + 1e-6)

            image_np = image.permute(1, 2, 0).numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image_np)
            axs[0].set_title("Input Image")
            axs[1].imshow(gt_vis, cmap='plasma')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_vis, cmap='plasma')
            axs[2].set_title("Predicted Depth")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"sample_{i}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[Saved] {save_path}")


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = Config()
    seed_everything(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"✅ Using device: {device}")
    full_dataset = DepthDataset(cfg.root_dir)
    train_idx, val_idx, test_idx = split_dataset(full_dataset, cfg)

    train_loader = DataLoader(DepthDataset(cfg.root_dir, train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(DepthDataset(cfg.root_dir, val_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(DepthDataset(cfg.root_dir, test_idx), batch_size=1, shuffle=False)

    model = DeepLabv3Plus_Depth().to(device)
    criterion = scale_invariant_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    start_epoch = 0
    resume = True
    last_ckpt = os.path.join(cfg.checkpoint_dir, "last.pth")

    # Resume 기능
    if resume and os.path.exists(last_ckpt):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, last_ckpt, device)
        start_epoch += 1
        print(f"[INFO] Resumed from epoch {start_epoch}, best loss {best_val_loss:.4f}")

    log_path = os.path.join(cfg.checkpoint_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    patience, no_improve = 5, 0

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n📘 Epoch {epoch+1}/{cfg.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{lr:.8f}\n")

        save_checkpoint(model, optimizer, epoch, val_loss, last_ckpt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, None, epoch, val_loss, os.path.join(cfg.checkpoint_dir, "best.pth"))
            print(f"💾 Best model updated ({best_val_loss:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping triggered!")
                break

    # Test phase
    print("\n🔍 Testing best model...")
    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pth")
    load_checkpoint(model, None, best_ckpt, device)
    test_loss = validate(model, test_loader, criterion, device)
    print(f"\n✅ Test loss: {test_loss:.4f}")

    # === 추가 ===
    visualize_predictions(model, train_dataset, device, num_samples=5, save_dir="./results_depth_train")
    visualize_predictions(model, val_dataset, device, num_samples=5, save_dir="./results_depth_val")
    visualize_predictions(model, test_dataset, device, num_samples=5, save_dir="./results_depth_test")