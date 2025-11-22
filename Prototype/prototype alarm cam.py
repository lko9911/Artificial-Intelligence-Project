import os
import random
import torch
import numpy as np
import cv2
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

checkpoint_path = "checkpoints_depth/best2.pth"

TARGET_SIZE = (512, 256)
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE

# =============================================================
# 거리 기반 색상
# =============================================================
def get_distance_color(distance, warning_threshold=7):
    if distance < warning_threshold:
        return (255, 0, 0)   # 빨강
    else:
        return (0, 255, 0)   # 초록

# =============================================================
# 카메라 내부 파라미터 (NYU 기준)
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
K_inv = np.linalg.inv(K_MATRIX)

# =============================================================
# YOLO 모델 로딩
# =============================================================
yolo_model = YOLO("YOLO/model/best.pt")

# =============================================================
# Depth 모델 로딩
# =============================================================
depth_model = DeepLabv3Plus_Depth(output_channels=1).to(device)

if not os.path.exists(checkpoint_path):
    print("[ERROR] 체크포인트 없음")
    sys.exit(0)

checkpoint = torch.load(checkpoint_path, map_location=device)
depth_model.load_state_dict(checkpoint["model_state"])
depth_model.eval()

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor()
])

# =============================================================
# 자동 웹캠 탐색
# =============================================================
def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[OK] 카메라 열림: 인덱스 {i}")
            return cap
        cap.release()
    return None

cap = find_working_camera()
if cap is None:
    print("[ERROR] 사용 가능한 웹캠을 찾을 수 없습니다.")
    sys.exit(0)

# 웹캠 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =============================================================
# 창 크기 고정 (1280 × 720)
# =============================================================
WINDOW_W, WINDOW_H = 1280, 720
cv2.namedWindow("YOLO + Depth 3D", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO + Depth 3D", WINDOW_W, WINDOW_H)

cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Map", WINDOW_W, WINDOW_H)

print("\n===== 실시간 추론 시작 (종료: Q) =====")

# =============================================================
# 실시간 루프
# =============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 웹캠 프레임 읽기 실패")
        break

    # PIL 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    # YOLO 추론
    yolo_results = yolo_model.predict(frame_rgb, save=False, conf=0.5, verbose=False)

    # 리사이즈
    resized_np = cv2.resize(frame_rgb, TARGET_SIZE)

    detected_objects = []

    if yolo_results and len(yolo_results[0].boxes) > 0:
        original_h, original_w = frame_rgb.shape[:2]
        scale_x = TARGET_WIDTH / original_w
        scale_y = TARGET_HEIGHT / original_h

        for box in yolo_results[0].boxes:
            x1o, y1o, x2o, y2o = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].item())
            class_name = yolo_results[0].names[class_id]

            x1 = int(x1o * scale_x)
            y1 = int(y1o * scale_y)
            x2 = int(x2o * scale_x)
            y2 = int(y2o * scale_y)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            detected_objects.append({
                'class': class_name,
                'center_2d': (center_x, center_y),
                'bbox_xyxy': (x1, y1, x2, y2),
                'center_3d': None
            })

    # Depth 예측
    depth_input = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_pred = depth_model(depth_input).squeeze().cpu().numpy().T

    depth_norm = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # ======= 3D 계산 & 화면 표시 =======
    display_img = resized_np.copy()

    for obj in detected_objects:
        cx, cy = obj['center_2d']
        cx = np.clip(cx, 0, TARGET_WIDTH - 1)
        cy = np.clip(cy, 0, TARGET_HEIGHT - 1)

        Z = depth_pred[cy, cx]

        pixel = np.array([cx, cy, 1])
        X, Y, Z_cam = (Z * (K_inv @ pixel))

        obj['center_3d'] = (X, Y, Z_cam)

        color = get_distance_color(Z_cam)

        x1, y1, x2, y2 = obj['bbox_xyxy']

        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_img,
                    f"{obj['class']} Z={Z_cam:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    # ======= 화면 출력 (고정 1280x720) =======
    resized_output = cv2.resize(display_img, (WINDOW_W, WINDOW_H))
    cv2.imshow("YOLO + Depth 3D", cv2.cvtColor(resized_output, cv2.COLOR_RGB2BGR))
    # ➕ 깊이맵 출력
    depth_resized = cv2.resize(depth_color, (WINDOW_W, WINDOW_H))
    cv2.imshow("Depth Map", depth_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
