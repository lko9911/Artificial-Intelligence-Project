import os
import json
import glob
from PIL import Image

SUNRGBD_ROOT = 'SUNRGBD'
OUTPUT_DIR = os.path.join(SUNRGBD_ROOT, 'yolo_bbox_output_safe') 
CLASSES_FILE = os.path.join(SUNRGBD_ROOT, 'classes.txt') 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. 클래스 이름 → ID 매핑
# -----------------------------
def build_class_mapping():
    class_mapping = {}
    with open(CLASSES_FILE, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                class_mapping[name] = i
    return class_mapping

# -----------------------------
# 2. JSON → YOLO bounding box 변환 (안전하게)
# -----------------------------
def convert_json_to_yolo_safe(json_files, class_mapping):
    for idx_file in json_files:
        print(f"처리 중: {os.path.basename(idx_file)}")
        try:
            with open(idx_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON 읽기 오류: {e}")
            continue

        # object id → 클래스 이름
        file_object_map = {}
        top_level_objects = data.get("objects", [])
        for i, obj in enumerate(top_level_objects):
            name = obj.get("name") if isinstance(obj, dict) else obj
            if name and name.strip():
                file_object_map[i] = name.strip()

        # 이미지 크기 확인
        img_dir = os.path.dirname(os.path.dirname(idx_file))
        img_files = glob.glob(os.path.join(img_dir, 'image', '*.jpg'))
        if not img_files:
            print(f"이미지 없음: {img_dir}")
            continue

        img_path = img_files[0]
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"이미지 열기 오류: {e}")
            continue

        # TXT 파일 경로
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        yolo_lines = []

        frames = data.get("frames", [])
        for frame in frames:
            polygons_to_process = frame.get("polygon", [frame])
            for p in polygons_to_process:
                obj_id = p.get("object")
                class_name = file_object_map.get(obj_id)
                if class_name is None:
                    continue

                x_list = p.get("x", [])
                y_list = p.get("y", [])
                if not x_list or not y_list or len(x_list) != len(y_list):
                    continue

                # 바운딩 박스 계산
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)

                # YOLO 정규화 & 안전 클리핑
                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # 0~1 범위로 클리핑
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                width = min(max(width, 0.0), 1.0)
                height = min(max(height, 0.0), 1.0)

                # width, height가 0이면 제거
                if width == 0 or height == 0:
                    continue

                class_id = class_mapping.get(class_name, -1)
                if class_id == -1:
                    continue

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(line)

        # 저장
        if yolo_lines:
            with open(txt_path, 'w') as f_out:
                f_out.write("\n".join(yolo_lines) + "\n")
            print(f"저장 완료: {txt_path} ({len(yolo_lines)}개 바운딩 박스)")

# -----------------------------
# 메인 실행
# -----------------------------
class_mapping = build_class_mapping()

search_pattern = os.path.join(SUNRGBD_ROOT, '*', 'NYUdata', '**', 'annotation', 'index.json')
json_files = glob.glob(search_pattern, recursive=True)

print(f"총 {len(json_files)}개의 JSON 파일 변환 시작")
convert_json_to_yolo_safe(json_files, class_mapping)
print("✅ 안전한 YOLO 바운딩 박스 변환 완료")
