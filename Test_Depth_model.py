import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DepthDataset
from model import DeepLabv3Plus_Depth
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터셋 및 모델 로드
test_dataset = DepthDataset("./Dataset")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = DeepLabv3Plus_Depth(output_channels=1).to(device)

# 체크포인트 로드
checkpoint = torch.load("./checkpoints_depth/best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

eps = 1e-6

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

        # -------------------------
        # 대표 지표 계산
        # -------------------------
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
print("=== Depth Evaluation Metrics ===")
print(f"Abs Rel: {np.mean(abs_rel_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"log10: {np.mean(log10_list):.4f}")
print(f"Delta1 (<1.25): {np.mean(delta1_list):.4f}")
