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

from dataset import DepthDataset
from model import DeepLabv3Plus_Depth


# =============================================================
# 기본 설정
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-6

source = "Depth/SUNRGBD/kv1/NYUdata/NYU1447/image/NYU1447.jpg"
checkpoint_path = "checkpoints_depth/best2.pth"

TARGET_SIZE = (512, 256)
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE


# =============================================================
# 거리 기반 색상 그라데이션
# =============================================================
def get_distance_color(distance, warning_threshold=7):
    """
    경고 거리 이하일 경우 빨강, 그 외는 초록
    """
    if distance < warning_threshold:
        return (255, 0, 0)  # 빨강
    else:
        return (0, 255, 0)  # 초록




# =============================================================
# 카메라 내부 파라미터 설정 (NYU 기준)
# =============================================================
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


# =============================================================
# YOLO 추론
# =============================================================
yolo_model = YOLO("YOLO/model/best.pt")
yolo_results = yolo_model.predict(source, save=False, conf=0.5, show=False)

detected_objects = []

if yolo_results and len(yolo_results[0].boxes) > 0:

    # 원본 이미지 로드 & 리사이즈
    image_raw = Image.open(source).convert('RGB')
    image_np_base = np.array(image_raw.resize(TARGET_SIZE))
    yolo_im_array_rgb = cv2.cvtColor(image_np_base, cv2.COLOR_RGB2BGR)

    original_width, original_height = yolo_results[0].orig_shape[1], yolo_results[0].orig_shape[0]
    scale_x = TARGET_WIDTH / original_width
    scale_y = TARGET_HEIGHT / original_height

    for box in yolo_results[0].boxes:
        x1o, y1o, x2o, y2o = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0].item())
        class_name = yolo_results[0].names[class_id]

        # 좌표 스케일링
        x1 = int(x1o * scale_x)
        y1 = int(y1o * scale_y)
        x2 = int(x2o * scale_x)
        y2 = int(y2o * scale_y)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # OpenCV 박스 (YOLO 텍스트는 사용 안함)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #cv2.rectangle(yolo_im_array_rgb, (x1, y1), (x2, y2), color, 2)

        detected_objects.append({
            'class': class_name,
            'center_2d': (center_x, center_y),
            'bbox_xyxy': (x1, y1, x2, y2),
            'center_3d': None,
            'conf': float(box.conf[0])
        })

    yolo_im_array_rgb = cv2.cvtColor(yolo_im_array_rgb, cv2.COLOR_BGR2RGB)

else:
    print("[WARNING] YOLO에서 객체가 감지되지 않았습니다.")
    sys.exit(0)


# =============================================================
# Depth 모델 로딩
# =============================================================
depth_model = DeepLabv3Plus_Depth(output_channels=1).to(device)

if not os.path.exists(checkpoint_path):
    print("[ERROR] 체크포인트 파일 없음")
    sys.exit(0)

checkpoint = torch.load(checkpoint_path, map_location=device)
depth_model.load_state_dict(checkpoint["model_state"])
depth_model.eval()

transform = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor()])
image_batch = transform(image_raw).unsqueeze(0).to(device)

with torch.no_grad():
    depth_pred_tensor = depth_model(image_batch)
    depth_pred_raw = depth_pred_tensor.squeeze().cpu().numpy()
    depth_pred = depth_pred_raw.T  # H x W


# =============================================================
# 3D 좌표 계산 + print 출력
# =============================================================
K_inv = np.linalg.inv(K_MATRIX)

print("\n========================================================")
print(f"| Calculated 3D Coordinates (X, Y, Z) in Meters |")
print("========================================================")

for obj in detected_objects:

    cx, cy = obj['center_2d']

    cy = np.clip(cy, 0, TARGET_HEIGHT - 1)
    cx = np.clip(cx, 0, TARGET_WIDTH - 1)

    Z = depth_pred[cy, cx]

    pixel_coords = np.array([cx, cy, 1.0])
    camera_coords = Z * (K_inv @ pixel_coords)

    X, Y, Z_final = camera_coords[0], camera_coords[1], camera_coords[2]

    obj['center_3d'] = (X, Y, Z_final)
    obj['distance'] = float(Z_final)

    warn = "⚠️ WARNING" if Z_final < 0.02 else ""

    print(f"| [{obj['class']:<10}] 2D=({cx:3d},{cy:3d}) | "
          f"3D=(X={X:6.2f}, Y={Y:6.2f}, Z={Z_final:6.2f}) m {warn}")

print("========================================================\n")


# =============================================================
# 시각화 (거리 기반 그라데이션 + WARNING 표시)
# =============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(yolo_im_array_rgb)

for obj in detected_objects:

    x1, y1, x2, y2 = obj['bbox_xyxy']
    Z = obj['distance']
    cx, cy = obj['center_2d']

    # 색상 그라데이션
    rgb = get_distance_color(Z)
    color = np.array(rgb) / 255.0

    # 박스
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # 클래스 표시
    ax.text(x1, y1 - 25, f"{obj['class']}",
            fontsize=9, color='white',
            bbox=dict(facecolor='blue', alpha=0.7))

    # Z + WARNING
    warn = " ⚠️ WARNING" if Z < 7 else ""
    ax.text(x1, y1 - 8, f"Z:{Z:.2f}m{warn}",
            fontsize=9, color='yellow',
            bbox=dict(facecolor='black', alpha=0.7))

ax.axis('off')
plt.tight_layout()
plt.show()
