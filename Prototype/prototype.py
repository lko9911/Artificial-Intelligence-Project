import os
import random
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from torchvision import transforms 
from PIL import Image 
import sys

# Depth Estimation 모델 관련 파일 (사용자 환경에 맞게 로드)
from dataset import DepthDataset 
from model import DeepLabv3Plus_Depth 

# ==========================
# 1. 기본 설정 및 경로
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-6

source = "Depth/SUNRGBD/kv1/NYUdata/NYU0005/image/NYU0005.jpg" 
checkpoint_path = "checkpoints_depth/best2.pth"

# DepthDataset의 img_size와 일치하는 크기 (width, height)
TARGET_SIZE = (512, 256) 
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE 

# ----------------------------------------------------
# 카메라 내부 매개변수 (K) 적용 및 스케일링 : NYU Depth 1번 사진 기준
# ----------------------------------------------------
FX_ORIGINAL = 518.857901
FY_ORIGINAL = 519.469611
CX_ORIGINAL = 284.582449
CY_ORIGINAL = 208.736166
ORIGINAL_WIDTH = 640.0
ORIGINAL_HEIGHT = 480.0

SCALE_X_K = TARGET_WIDTH / ORIGINAL_WIDTH
SCALE_Y_K = TARGET_HEIGHT / ORIGINAL_HEIGHT

FX = FX_ORIGINAL * SCALE_X_K
FY = FY_ORIGINAL * SCALE_Y_K
CX = CX_ORIGINAL * SCALE_X_K
CY = CY_ORIGINAL * SCALE_Y_K

K_MATRIX = np.array([
    [FX, 0, CX], 
    [0, FY, CY], 
    [0, 0, 1]
])
# ----------------------------------------------------


# ==========================
# 2. YOLO 모델 설정 및 추론
# ==========================
yolo_model = YOLO("YOLO/model/best.pt")

# plot=False로 설정하여 YOLO 자체 텍스트를 그리지 않고, 우리가 직접 그릴 수 있도록 준비합니다.
# 하지만 YOLO의 plot()을 사용하는 이유는 바운딩 박스를 쉽게 얻기 위함이므로,
# 여기서는 plot()을 사용하되, 나중에 텍스트를 덮어씌우는 방식으로 진행하겠습니다.
yolo_results = yolo_model.predict(source, save=False, conf=0.5, show=False)
detected_objects_3d = [] 

if yolo_results and len(yolo_results[0].boxes) > 0:
    # --------------------------------------------------------------------------
    # YOLO의 plot()을 사용하지 않고, 우리가 직접 바운딩 박스를 그린 배경 이미지를 만듭니다.
    # 이렇게 해야 YOLO가 자동으로 그린 Confidence 텍스트를 완전히 제어할 수 있습니다.
    # --------------------------------------------------------------------------
    image_raw = Image.open(source).convert('RGB')
    image_np_base = np.array(image_raw.resize(TARGET_SIZE))
    yolo_im_array_rgb = cv2.cvtColor(image_np_base, cv2.COLOR_RGB2BGR) # OpenCV용 BGR로 변환
    
    original_width, original_height = yolo_results[0].orig_shape[1], yolo_results[0].orig_shape[0]
    scale_x = TARGET_WIDTH / original_width
    scale_y = TARGET_HEIGHT / original_height
    
    for box in yolo_results[0].boxes:
        x1_orig, y1_orig, x2_orig, y2_orig = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0].item())
        class_name = yolo_results[0].names[class_id]
        conf = box.conf[0].item()
        
        # 리사이즈된 이미지에서의 좌표
        x1 = int(x1_orig * scale_x)
        y1 = int(y1_orig * scale_y)
        x2 = int(x2_orig * scale_x)
        y2 = int(y2_orig * scale_y)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # ----------------------------------------------------------------------
        # 우리가 직접 바운딩 박스와 텍스트를 그립니다. (OpenCV 사용)
        # ----------------------------------------------------------------------
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # BGR
        cv2.rectangle(yolo_im_array_rgb, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트는 Depth 계산 후에 최종적으로 Matplotlib으로 그릴 것입니다.
        # YOLO가 기본적으로 표시하는 텍스트는 여기서 생략합니다.
        
        detected_objects_3d.append({
            'class': class_name,
            'center_2d': (center_x, center_y),
            'center_3d': None, 
            'bbox_start': (x1, y1), # 텍스트를 그릴 위치를 위해 저장
            'conf': conf
        })
    
    # Matplotlib 출력을 위해 다시 RGB로 변환
    yolo_im_array_rgb = cv2.cvtColor(yolo_im_array_rgb, cv2.COLOR_BGR2RGB)
        
else:
    print("[WARNING] YOLO Prediction failed or no object detected. Exiting.")
    sys.exit(0) 


# ==========================
# 3. Depth Estimation 모델 설정 및 추론 (이전과 동일)
# ==========================
depth_model = DeepLabv3Plus_Depth(output_channels=1).to(device)
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    depth_model.load_state_dict(checkpoint["model_state"])
depth_model.eval()

try:
    image_raw = Image.open(source).convert('RGB')
    transform = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor()])
    image_batch = transform(image_raw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth_pred_tensor = depth_model(image_batch)
        depth_pred_raw = depth_pred_tensor.squeeze().cpu().numpy()
        depth_pred = depth_pred_raw.T 
        
    K_inv = np.linalg.inv(K_MATRIX)
    
    print("\n========================================================")
    print(f"| Calculated 3D Coordinates (X, Y, Z) in Meters for {os.path.basename(source)} |")
    print("========================================================")

    for obj in detected_objects_3d:
        cx, cy = obj['center_2d']
        
        cy_safe = np.clip(cy, 0, TARGET_HEIGHT - 1)
        cx_safe = np.clip(cx, 0, TARGET_WIDTH - 1)
        Z = depth_pred[cy_safe, cx_safe]
        
        pixel_coords = np.array([cx, cy, 1.0])
        camera_coords = Z * K_inv @ pixel_coords
        X, Y, Z_final = camera_coords[0], camera_coords[1], camera_coords[2] 
        
        obj['center_3d'] = (X, Y, Z_final)
        
        print(f"| [{obj['class']:<10}] 2D: ({cx:3d}, {cy:3d}) | 3D: (X={X:5.2f}, Y={Y:5.2f}, Z={Z_final:5.2f}) m |")
    print("========================================================")

except Exception as e:
    print(f"\n[ERROR] Depth 추론 또는 3D 좌표 계산 중 오류 발생: {e}")
    sys.exit(0)


# ==========================
# 4. 최종 결과 시각화 (YOLO Detection + Z-Depth 단일 그림)
# ==========================
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5)) 
ax.imshow(yolo_im_array_rgb, aspect='equal') 
#ax.set_title(f"YOLO Detection with Object Depth (Z in Meters)", fontsize=14)


for obj in detected_objects_3d:
    cx, cy = obj['center_2d']
    X, Y, Z = obj['center_3d']
    x1, y1 = obj['bbox_start']
    
    # 1. 2D 중간점 표시 (선택 사항)
    # ax.plot(cx, cy, 'o', color='red', markersize=5, markerfacecolor='yellow', markeredgecolor='red') 
    
    # 2. 바운딩 박스 위에 클래스 이름과 Depth (Z) 값 표시
    # 텍스트 위치를 바운딩 박스의 시작점 (x1, y1) 근처로 설정
    # 클래스 이름 텍스트
    class_label = f"{obj['class']}" 
    ax.text(x1, y1 - 25, class_label, color='white', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Depth Z 값 텍스트 (Confidence를 대체하는 위치)
    depth_label = f"Z:{Z:.2f}m" 
    ax.text(x1, y1 - 8, depth_label, color='yellow', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')) 

ax.axis('off')
plt.tight_layout()
plt.show()