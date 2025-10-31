import os
import random
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from torchvision import transforms 
from PIL import Image 

# Depth Estimation 모델 관련 파일 (사용자 환경에 맞게 로드)
# 이 파일들은 동일한 디렉토리에 있어야 합니다.
from dataset import DepthDataset 
from model import DeepLabv3Plus_Depth 

# ==========================
# 1. 기본 설정 및 경로
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-6

# 단일 이미지 경로
source = "Depth/SUNRGBD/kv1/NYUdata/NYU0002/image/NYU0002.jpg" 

# Depth 모델의 체크포인트 경로
checkpoint_path = "Depth/checkpoints_depth/best.pth"

# DepthDataset의 img_size와 일치하는 크기 (width, height)
TARGET_SIZE = (512, 256) 
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE 


# ==========================
# 2. YOLO 모델 설정 및 추론
# ==========================
print(f"[{os.path.basename(__file__)}] Starting analysis...")
yolo_model = YOLO("YOLO/model/best.pt")

# YOLO 추론 실행
yolo_results = yolo_model.predict(source, save=False, conf=0.5, show=False)

if yolo_results:
    yolo_im_array_bgr_original = yolo_results[0].plot() 
    
    # YOLO 결과물을 TARGET_SIZE로 리사이즈하여 크기 통일
    yolo_im_array_bgr_resized = cv2.resize(yolo_im_array_bgr_original, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    yolo_im_array_rgb = cv2.cvtColor(yolo_im_array_bgr_resized, cv2.COLOR_BGR2RGB)
    print(f"[DEBUG] YOLO Output Shape: {yolo_im_array_rgb.shape}")
else:
    print("[WARNING] YOLO Prediction failed or no object detected.")
    yolo_im_array_rgb = None


# ==========================
# 3. Depth Estimation 모델 설정 및 추론
# ==========================
depth_model = DeepLabv3Plus_Depth(output_channels=1).to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    depth_model.load_state_dict(checkpoint["model_state"])
    print(f"[INFO] Loaded Depth checkpoint: {checkpoint_path}")
else:
    print("[WARNING] Depth checkpoint not found. Using untrained model.")

depth_model.eval()

# --- 입력 이미지 및 GT Depth 맵 로드 및 전처리 ---

try:
    # 1. Depth GT 맵 로드 및 전처리 (DepthDataset의 로직과 동일: Normalize 없음)
    depth_gt_source = source.replace("image", "depth").replace(".jpg", ".png")
    depth_gt_map_raw = cv2.imread(depth_gt_source, cv2.IMREAD_UNCHANGED).astype("float32") / 1000.0
    depth_gt_map_raw[depth_gt_map_raw <= 0] = 0.01

    # 2. RGB 입력 이미지 로드 및 전처리
    image_raw = Image.open(source).convert('RGB')
    
    # DepthDataset의 전처리: Resize(PIL) -> ToTensor() 만 적용
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), 
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image_raw)
    
    # GT Depth 맵 전처리 
    depth_gt_resized = cv2.resize(depth_gt_map_raw, TARGET_SIZE) 
    depth_gt_tensor = torch.from_numpy(depth_gt_resized).unsqueeze(0)
    
    # 추론 준비
    image_batch = image_tensor.unsqueeze(0).to(device)
    depth_gt = depth_gt_tensor.squeeze().numpy()
    
    # 시각화용 이미지 
    image_np_display = np.array(image_raw.resize(TARGET_SIZE)) / 255.0
    
    # 3. Depth 추론
    with torch.no_grad():
        depth_pred_tensor = depth_model(image_batch)
        depth_pred_raw = depth_pred_tensor.squeeze().cpu().numpy()
        
        # ★★★ 문제 해결: Height x Width 순서로 맞추기 위해 전치(Transpose) 적용 ★★★
        depth_pred = depth_pred_raw.T 
        print(f"[DEBUG] Predicted Depth Shape (AFTER TRANSPOSE): {depth_pred.shape}")


    # 4. 시각화를 위한 깊이 정규화 
    def normalize_depth(depth_map):
        mask = depth_map > 0
        min_d = np.min(depth_map[mask]) if np.any(mask) else 0
        max_d = np.max(depth_map[mask]) if np.any(mask) else 1
        return (depth_map - min_d) / (max_d - min_d + 1e-6)

    pred_vis = normalize_depth(depth_pred) 
    gt_vis = normalize_depth(depth_gt) 
    
    print(f"[DEBUG] Input Image Display Shape: {image_np_display.shape}")
    print(f"[DEBUG] GT Depth Shape: {gt_vis.shape}")
    
except Exception as e:
    print(f"[ERROR] Depth 추론 또는 로드 중 오류 발생: {e}")
    is_depth_ok = False
else:
    is_depth_ok = True


# ==========================
# 4. 통합 결과 시각화
# ==========================
if yolo_im_array_rgb is not None and is_depth_ok:
    # 4개 모두 출력
    fig, axs = plt.subplots(1, 4, figsize=(18, 4)) 
    
    # 모든 이미지에 aspect='equal'을 적용하여 왜곡 방지
    
    # (1) 입력 이미지 
    axs[0].imshow(image_np_display, aspect='equal') 
    axs[0].set_title("Input Image", fontsize=12)
    
    # (2) YOLO 예측 결과 
    axs[1].imshow(yolo_im_array_rgb, aspect='equal') 
    axs[1].set_title(f"YOLO Detection (Conf > {0.5})", fontsize=12)

    # (3) Ground Truth Depth 
    axs[2].imshow(gt_vis, cmap='plasma', aspect='equal') 
    axs[2].set_title("Ground Truth Depth", fontsize=12)
    
    # (4) Predicted Depth (이제 순서가 올바름)
    axs[3].imshow(pred_vis, cmap='plasma', aspect='equal') 
    axs[3].set_title("Predicted Depth", fontsize=12)
    

    # 축 정보 제거
    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"Integrated Analysis for: {os.path.basename(source)} (Resized to {TARGET_WIDTH}x{TARGET_HEIGHT})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("\n[FAIL] 시각화 실패: YOLO 또는 Depth 모델 추론에 필요한 모든 요소가 준비되지 않았습니다.")