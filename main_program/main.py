import sys
import os
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QDialog, QComboBox, QSpinBox, QFormLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from model import DeepLabv3Plus_Depth

# =============================================================
# 카메라 내부 파라미터 (NYU 기준)
# =============================================================

TARGET_SIZE = (512, 256)
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE


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

# ================================================================
# 커스텀 페이지
# ================================================================
class CustomSettingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Settings")
        self.setFixedSize(350, 250)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        self.color_mode = QComboBox()
        self.color_mode.addItems(["Normal", "Protanopia", "Deuteranopia", "Tritanopia"])

        self.dist_threshold = QSpinBox()
        self.dist_threshold.setRange(1, 30)
        self.dist_threshold.setValue(7)

        form = QFormLayout()
        form.addRow("Box Gradient Mode:", self.color_mode)
        form.addRow("Distance Warning Threshold (m):", self.dist_threshold)

        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        h_btn = QHBoxLayout()
        h_btn.addWidget(btn_ok)
        h_btn.addWidget(btn_cancel)

        v_layout = QVBoxLayout()
        v_layout.addLayout(form)
        v_layout.addLayout(h_btn)
        self.setLayout(v_layout)

    def get_settings(self):
        return {"color_mode": self.color_mode.currentText(), "threshold": self.dist_threshold.value()}


# ================================================================
# Main GUI - 이미지/텍스트/실행부/버튼
# ================================================================
class DepthYoloGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prototype Viewer - Developed by 이규원")
        self.setGeometry(50, 50, 800, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        main_widget = QWidget()
        h_layout = QHBoxLayout(main_widget)

        # 이미지
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(400, 200)
        if os.path.exists("main_program/image/developer.png"):
            pix = QPixmap("main_program/image/developer.png").scaled(400, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(pix)
        else:
            self.image_label.setText("Main Image")

        # 개발자 문구
        self.text_label = QLabel("<인공지능 C팀>\n\n\n 저시력자를 위한 실내 근거리 물체 탐지 및 알림 시스템\n팀원: 이규원, 신재호, 박성진, 최윤혁, 김지호")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("font-size: 18px; color: white;")

        h_layout.addWidget(self.image_label)
        h_layout.addWidget(self.text_label)

        # 로그창
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: white;")

        # 버튼
        self.btn_custom = QPushButton("Custom Setting")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")

        # 레이아웃
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_custom)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        main_layout = QVBoxLayout()
        main_layout.addWidget(main_widget) 
        main_layout.addWidget(self.log_box)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # =========================================================
        # YOLO + Depth
        # =========================================================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO("main_program/YOLO/model/best.pt")
        self.depth_model = self.load_depth_model()
        self.transform = transforms.Compose([transforms.Resize((512, 256)), transforms.ToTensor()])

        self.K_inv = K_inv
        self.cap = None
        self.running = False
        self.color_mode = "Normal"
        self.distance_threshold = 7

        # 버튼 이벤트
        self.btn_custom.clicked.connect(self.open_custom_setting)
        self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop.clicked.connect(self.stop_cam)

    # =========================================================
    # 거리 기반 박스 색상 반환 (color_mode + distance_threshold 반영)
    # =========================================================
    def get_distance_color(self, Z):
        """
        Z 값과 self.color_mode에 따라 색상을 반환합니다.
        - 가까운 물체(Z < self.distance_threshold) : 빨강 계열
        - 먼 물체(Z >= self.distance_threshold) : 초록 계열
        - color_mode에 따라 색상 톤 변경 가능
        """
        if Z < self.distance_threshold:
            # 가까운 물체
            if self.color_mode == "Normal":
                return (0, 0, 255)  # 빨강
            elif self.color_mode == "Protanopia":
                return (0, 128, 255)
            elif self.color_mode == "Deuteranopia":
                return (255, 128, 0)
            elif self.color_mode == "Tritanopia":
                return (255, 0, 128)
        else:
            # 먼 물체
            if self.color_mode == "Normal":
                return (0, 255, 0)  # 초록
            elif self.color_mode == "Protanopia":
                return (128, 255, 128)
            elif self.color_mode == "Deuteranopia":
                return (0, 255, 255)
            elif self.color_mode == "Tritanopia":
                return (128, 128, 255)

    # =========================================================
    def log(self, text):
        self.log_box.append(text)
        print(text)
        

    # =========================================================
    def load_depth_model(self):
        model = DeepLabv3Plus_Depth(output_channels=1).to(self.device)
        ckpt = "main_program/Depth/model/best.pth"
        if not os.path.exists(ckpt):
            self.log("[ERROR] Depth checkpoint not found.")
            return model
        data = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(data["model_state"])
        model.eval()
        self.log("[OK] Depth model loaded")
        return model

    # =========================================================
    def open_custom_setting(self):
        dlg = CustomSettingDialog()
        if dlg.exec():
            settings = dlg.get_settings()
            self.color_mode = settings["color_mode"]
            self.distance_threshold = settings["threshold"]
            self.log(f"[SET] Color Mode: {self.color_mode}, Threshold: {self.distance_threshold}m")

    # =========================================================
    def start_cam(self):
        if self.cap is None:
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.cap = cap
                    self.log(f"[OK] Camera opened index {i}")
                    break

        if not self.cap:
            self.log("[ERROR] No camera detected")
            return

        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("[ERROR] Failed to read webcam frame")
                break

            # PIL 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            # YOLO 추론
            yolo_results = self.yolo_model.predict(frame_rgb, save=False, conf=0.5, verbose=False)

            # 리사이즈
            resized_np = cv2.resize(frame_rgb, (TARGET_WIDTH, TARGET_HEIGHT))

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
            depth_input = self.transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                depth_pred = self.depth_model(depth_input).squeeze().cpu().numpy().T

            # Depth 후처리
            depth_norm = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_norm = depth_norm.astype(np.uint8)
            #depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            # 3D 계산 & 화면 표시
            display_img = resized_np.copy()

            for obj in detected_objects:
                cx, cy = obj['center_2d']
                cx = np.clip(cx, 0, TARGET_WIDTH - 1)
                cy = np.clip(cy, 0, TARGET_HEIGHT - 1)

                Z = depth_pred[cy, cx]

                # 카메라 좌표계 변환 (K_inv 필요)
                pixel = np.array([cx, cy, 1])
                X, Y, Z_cam = (Z * (K_inv @ pixel))
                
                obj['center_3d'] = (X, Y, Z_cam)

                color = self.get_distance_color(Z_cam)  # 거리 기반 색상

                x1, y1, x2, y2 = obj['bbox_xyxy']
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                
                #cv2.putText(display_img, f"{obj['class']} Z={Z_cam:.2f}",
                #            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                def draw_text_custom(img, text, pos, text_color, bg_color, font_scale=0.5, thickness=1):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x, y = pos
                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + 2), bg_color, -1)
                    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

                draw_text_custom(display_img, obj['class'], (x1, y1 - 25), (255, 255, 255), (0, 0, 255))

                
                warn_text = " WARNING" if Z_cam < self.distance_threshold else ""
                info_text = f"Z:{Z:.2f}m{warn_text}"
                draw_text_custom(display_img, info_text, (x1, y1 - 8), (255, 255, 0), (0, 0, 0))

            # ======= 화면 출력 (고정 1280x720) =======
            WINDOW_W, WINDOW_H = 1280, 720
            resized_output = cv2.resize(display_img, (WINDOW_W, WINDOW_H))
            cv2.imshow("YOLO + Depth 3D", cv2.cvtColor(resized_output, cv2.COLOR_RGB2BGR))

            for obj in detected_objects:
                self.log(f"[DETECTED] Class: {obj['class']}, Z={Z_cam:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def stop_cam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.log("[STOP] Prototype View closed")

    # =========================================================
    def get_box_color(self):
        if self.color_mode == "Normal":
            return (0, 255, 0)
        if self.color_mode == "Protanopia":
            return (0, 128, 255)
        if self.color_mode == "Deuteranopia":
            return (255, 128, 0)
        if self.color_mode == "Tritanopia":
            return (255, 0, 128)
        return (0, 255, 0)


# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DepthYoloGUI()
    gui.show()
    sys.exit(app.exec())
