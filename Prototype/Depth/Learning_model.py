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
    root_dir = "./Dataset"
    checkpoint_dir = "./checkpoints_depth"
    batch_size = 4
    learning_rate = 1e-4
    num_workers = 4
    epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 42


# ==========================
# DepthDataset 수정
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
        # --- Image ---
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size)
        image = transforms.ToTensor()(image)  # 0~1 범위, (C,H,W)

        # --- Depth ---
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32") / 1000.0
        depth[depth <= 0] = 0.01
        depth = cv2.resize(depth, self.img_size)  # OpenCV로 resize
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1,H,W)

        return image, depth

# ==========================
# Scale-Invariant Loss 수정
# ==========================
def scale_invariant_loss(pred, target):
    # valid_mask 만들기
    valid_mask = (target > 0)
    
    # pred와 target 크기 맞추기
    if pred.shape != target.shape:
        target = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='nearest')
        valid_mask = (target > 0)

    # pred와 target에 valid_mask 적용
    pred = pred[valid_mask]
    target = target[valid_mask]

    # 만약 valid 픽셀이 없다면 0 반환
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # NaN 방지용 clamp
    pred = torch.clamp(pred, min=0.01)
    target = torch.clamp(target, min=1e-3)

    # 로그 계산
    diff = torch.log(torch.clamp(pred, min=0.01)) - torch.log(torch.clamp(target, min=0.01))
    loss = torch.mean(diff ** 2) - (torch.mean(diff) ** 2)

    # NaN 체크
    if torch.isnan(loss):
        print("[WARNING] Loss is NaN!")
        print("pred min/max:", pred.min().item(), pred.max().item())
        print("target min/max:", target.min().item(), target.max().item())
    
    return loss


# ==========================
# 모델 정의
# ==========================
class DeepLabv3Plus_Depth(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[-1] = nn.Sequential(
    nn.Conv2d(256, 1, kernel_size=1),
    nn.ReLU()  # 음수 방지
)

    def forward(self, x):
        return self.model(x)["out"]


# ==========================
# 손실 함수 (Scale-Invariant)
# ==========================
def scale_invariant_loss(pred, target):
    valid_mask = (target > 0)
    if pred.shape != target.shape:
        target = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='nearest')
        valid_mask = (target > 0)
    pred = pred[valid_mask]
    target = target[valid_mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
    return torch.mean(diff ** 2) - (torch.mean(diff) ** 2)


# ==========================
# 학습 및 검증 함수
# ==========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, depths in tqdm(dataloader, desc="Training", ncols=100):
        imgs, depths = imgs.to(device), depths.to(device)

        preds = model(imgs)
        
        # --- 디버그용 출력 ---
        if torch.isnan(preds).any():
            print("[WARNING] NaN in preds!")
        if torch.isnan(depths).any():
            print("[WARNING] NaN in depths!")
        #print("Pred min/max:", preds.min().item(), preds.max().item())
        #print("Target min/max:", depths.min().item(), depths.max().item())
        # ----------------------

        loss = criterion(preds, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch completed | Average Loss: {avg_loss:.4f}")
    return avg_loss



def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, depths in loader:
            images, depths = images.to(device), depths.to(device)
            preds = model(images)
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
# 데이터셋 분할 함수
# ==========================
def split_dataset(dataset, cfg):
    total_size = len(dataset)
    train_size = int(total_size * cfg.train_ratio)
    val_size = int(total_size * cfg.val_ratio)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    random.seed(cfg.seed)
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices

# ==========================
# Main Loop
# ==========================
if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    cfg = Config()
    device = cfg.device

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # 전체 데이터셋 로드 및 분할
    full_dataset = DepthDataset(root_dir=cfg.root_dir)
    train_idx, val_idx, test_idx = split_dataset(full_dataset, cfg)

    train_dataset = DepthDataset(cfg.root_dir, indices=train_idx)
    val_dataset = DepthDataset(cfg.root_dir, indices=val_idx)
    test_dataset = DepthDataset(cfg.root_dir, indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = DeepLabv3Plus_Depth(output_channels=1).to(device)
    criterion = scale_invariant_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    start_epoch = 0
    resume = True
    last_ckpt = os.path.join(cfg.checkpoint_dir, "last.pth")

    '''
    # -----------------------------
    # 1~3번 데이터 확인 코드
    # -----------------------------
    print("=== 데이터 확인 ===")
    # 첫 샘플
    image, depth = train_dataset[0]
    print("Depth min:", depth.min().item(), " max:", depth.max().item(), " mean:", depth.mean().item())
    print("Image min:", image.min().item(), " max:", image.max().item(), " mean:", image.mean().item())

    # 배치 단위 확인
    for imgs, depths in train_loader:
        print("Batch imgs min/max:", imgs.min().item(), imgs.max().item())
        print("Batch depths min/max:", depths.min().item(), depths.max().item())
        break  # 첫 배치만 확인

    # 시각화 (선택)
    import matplotlib.pyplot as plt
    img_np = image.permute(1, 2, 0).numpy()
    depth_np = depth.squeeze().numpy()
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.title("Image")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(depth_np, cmap='plasma')
    plt.title("Depth")
    plt.axis('off')
    plt.show()
    # -----------------------------
    '''
    # 체크포인트 불러오기
    if resume and os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, last_ckpt, device)
        start_epoch += 1
        print(f"[INFO] Start from epoch {start_epoch}, best loss: {best_val_loss:.4f}")

    log_path = os.path.join(cfg.checkpoint_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")
    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch+1}/{cfg.epochs}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

        # 로그 기록 추가
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{lr:.8f}\n")

        save_checkpoint(model, optimizer, epoch, val_loss, last_ckpt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f">>> Best model updated! ({best_val_loss:.4f})")
            save_checkpoint(model, None, epoch, best_val_loss, os.path.join(cfg.checkpoint_dir, "best.pth"))

    # ==========================
    # Test & Visualization
    # ==========================
    def visualize_predictions(model, dataset, device, num_samples=5, save_dir="./results_depth"):
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        indices = random.sample(range(len(dataset)), num_samples)
        with torch.no_grad():
            for i in indices:
                image, depth_gt = dataset[i]
                image_batch = image.unsqueeze(0).to(device)
                pred = model(image_batch)
                pred = pred.squeeze().cpu().numpy()

                # 깊이 정규화 (시각화용)
                pred_vis = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-6)
                gt_vis = depth_gt.squeeze().numpy()
                gt_vis = (gt_vis - np.min(gt_vis)) / (np.max(gt_vis) - np.min(gt_vis) + 1e-6)

                image_np = image.permute(1, 2, 0).numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(image_np)
                axs[0].set_title("Input Image")
                axs[1].imshow(gt_vis, cmap='plasma')
                axs[1].set_title("Ground Truth Depth")
                axs[2].imshow(pred_vis, cmap='plasma')
                axs[2].set_title("Predicted Depth")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f"sample_{i}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"[Saved] {save_path}")

    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pth")
    if os.path.exists(best_ckpt):
        print(f"[INFO] Loading best model from {best_ckpt}")
        load_checkpoint(model, None, best_ckpt, device)
    else:
        print("[WARNING] Best checkpoint not found. Using last model.")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loss = validate(model, test_loader, criterion, device)
    print(f"\n[Test Result] Average test loss: {test_loss:.4f}")

    visualize_predictions(model, test_dataset, device, num_samples=5)
    