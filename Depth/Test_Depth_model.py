import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import DepthDataset
from model import DeepLabv3Plus_Depth

# ==========================
# 기본 설정
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-6
checkpoint_path = "./checkpoints_depth/best2.pth"

# ==========================
# 데이터셋 및 모델 로드
# ==========================
test_dataset = DepthDataset("Depth/Dataset")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = DeepLabv3Plus_Depth(output_channels=1).to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
else:
    print("[WARNING] Checkpoint not found. Using untrained model.")

model.eval()

# ==========================
# 평가 지표 계산
# ==========================
abs_rel_list, rmse_list, log10_list, delta1_list = [], [], [], []

with torch.no_grad():
    for images, depths in tqdm(test_loader, desc="Testing", ncols=100):
        images, depths = images.to(device), depths.to(device)
        preds = model(images)

        pred = preds.squeeze().cpu().numpy()
        target = depths.squeeze().cpu().numpy()

        # divide by zero 방지
        pred = np.clip(pred, eps, None)
        target = np.clip(target, eps, None)

        abs_rel = np.mean(np.abs(target - pred) / target)
        rmse = np.sqrt(np.mean((target - pred) ** 2))
        log10 = np.mean(np.abs(np.log10(target) - np.log10(pred)))

        thresh = np.maximum(pred / target, target / pred)
        delta1 = (thresh < 1.25).mean()

        abs_rel_list.append(abs_rel)
        rmse_list.append(rmse)
        log10_list.append(log10)
        delta1_list.append(delta1)

# 평균 지표 출력
print("\n=== Depth Evaluation Metrics ===")
print(f"Abs Rel: {np.mean(abs_rel_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"log10: {np.mean(log10_list):.4f}")
print(f"Delta1 (<1.25): {np.mean(delta1_list):.4f}")

# ==========================
# 결과 시각화 함수
# ==========================
def visualize_predictions(model, dataset, device, num_samples=5, save_dir="./results_depth2"):
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

# ==========================
# 시각화 실행
# ==========================
visualize_predictions(model, test_dataset, device, num_samples=5)
