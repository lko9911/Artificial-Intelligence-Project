import os
import random
import torch
import numpy as np
import cv2
import sys
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# dataset, model 파일이 같은 경로에 있다고 가정
from dataset import DepthDataset
from model import DeepLabv3Plus_Depth

# =============================================================
# 1. GPU 설정 확인 (가장 중요)
# =============================================================
device = "cuda"
print(f"Running on: {device}")  # 현재 장치 확인용 출력

source = "images/3.jpg"
checkpoint_path = "models/depths.pth"

TARGET_SIZE = (640, 480)
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE

# =============================================================
# 거리 기반 색상 (변경 없음)
# =============================================================
def get_distance_color(distance, warning_threshold=7):
    if distance < warning_threshold:
        return (255, 0, 0)
    else:
        return (0, 255, 0)

# =============================================================
# 카메라 파라미터 (NumPy 유지 - 좌표 계산용)
# =============================================================
# 좌표 계산은 행렬 연산이 가벼워 CPU(NumPy)가 오히려 빠를 수 있습니다.
# 굳이 Tensor로 바꿔 GPU로 보내면 전송 오버헤드가 더 큽니다.
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
K_inv = np.linalg.inv(K_MATRIX) # 미리 계산해둠

# =============================================================
# 모델 로딩 (GPU로 이동)
# =============================================================
print("Loading Models...")
# 1. YOLO 모델 로드
yolo_model = YOLO("models/detect.pt")

# 2. Depth 모델 로드 및 GPU 이동
depth_model = DeepLabv3Plus_Depth(output_channels=1).to(device)
if not os.path.exists(checkpoint_path):
    print("[ERROR] 체크포인트 파일 없음")
    sys.exit(0)

checkpoint = torch.load(checkpoint_path, map_location=device)
depth_model.load_state_dict(checkpoint["model_state"])
depth_model.eval()

# 전처리용 Transform (GPU에서 처리하기 위해 ToTensor만 사용)
# Resize는 cv2에서 미리 수행함
transform_to_tensor = transforms.ToTensor()

def detecting(image):
    # =============================================================
    # 1. 이미지 전처리
    # =============================================================
    # 입력 이미지를 미리 리사이즈 (CPU 작업)
    input_frame_bgr = cv2.resize(image, TARGET_SIZE)
    display_image = input_frame_bgr.copy()

    # Depth 모델 입력을 위한 RGB 변환 및 Tensor 변환
    input_frame_rgb = cv2.cvtColor(input_frame_bgr, cv2.COLOR_BGR2RGB)
    
    # PIL 변환 없이 바로 Tensor로 변환 후 GPU로 전송 (속도 향상)
    # image_batch: (1, 3, 480, 640) on GPU
    image_batch = transform_to_tensor(input_frame_rgb).unsqueeze(0).to(device)

    # =============================================================
    # 2. YOLO 추론 (GPU 강제 설정)
    # =============================================================
    # device=device 파라미터를 넣어 확실하게 GPU 사용
    yolo_results = yolo_model.predict(input_frame_bgr, save=False, conf=0.5, show=False, verbose=False, device=device)

    detected_objects = []

    if not yolo_results or len(yolo_results[0].boxes) == 0:
        return display_image

    # =============================================================
    # 3. Depth 모델 추론 (GPU)
    # =============================================================
    with torch.no_grad():
        depth_pred_tensor = depth_model(image_batch)
        # 결과: (1, 1, 480, 640) -> squeeze -> (480, 640)
        # 여기서 바로 .cpu().numpy()를 하지 않고 필요한 픽셀만 가져올 수도 있지만,
        # 전체 맵이 필요하므로 CPU로 내립니다.
        depth_pred = depth_pred_tensor.squeeze().cpu().numpy()

    # =============================================================
    # 4. 데이터 처리 및 좌표 계산 (CPU)
    # =============================================================
    # boxes 데이터 추출 (GPU Tensor -> CPU NumPy)
    boxes = yolo_results[0].boxes
    boxes_cpu = boxes.xyxy.cpu().numpy().astype(int)
    clss_cpu = boxes.cls.cpu().numpy().astype(int)
    confs_cpu = boxes.conf.cpu().numpy()
    names = yolo_results[0].names

    for i in range(len(boxes_cpu)):
        x1, y1, x2, y2 = boxes_cpu[i]
        class_id = clss_cpu[i]
        class_name = names[class_id]
        conf = confs_cpu[i]

        # 중심점
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 좌표 클리핑 (Index Error 방지)
        center_x = np.clip(center_x, 0, TARGET_WIDTH - 1)
        center_y = np.clip(center_y, 0, TARGET_HEIGHT - 1)

        # Depth 값 추출
        Z = depth_pred[center_y, center_x]

        # 3D 좌표 계산 (행렬 연산)
        pixel_coords = np.array([center_x, center_y, 1.0])
        camera_coords = Z * (K_inv @ pixel_coords)
        X, Y, Z_final = camera_coords

        detected_objects.append({
            'bbox': (x1, y1, x2, y2),
            'class': class_name,
            'distance': float(Z_final)
        })

    # =============================================================
    # 5. 시각화 (CPU - OpenCV)
    # =============================================================
    # OpenCV 그리기 함수는 CPU 메모리에 있는 numpy array만 취급합니다.
    
    def draw_text_custom(img, text, pos, text_color, bg_color, font_scale=0.5, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = pos
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + 2), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        Z = obj['distance']
        class_name = obj['class']
        
        rgb = get_distance_color(Z)
        color_bgr = rgb[::-1]
        
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color_bgr, 2)
        
        draw_text_custom(display_image, class_name, (x1, y1 - 25), (255, 255, 255), (255, 0, 0))
        
        warn_text = " WARNING" if Z < 7 else ""
        info_text = f"Z:{Z:.2f}m{warn_text}"
        draw_text_custom(display_image, info_text, (x1, y1 - 8), (0, 255, 255), (0, 0, 0))

    return display_image