import os
import json
import glob
from PIL import Image

# --- 설정 (CONFIG) ---
SUNRGBD_ROOT = 'SUNRGBD'
# 경고: 이 폴더의 파일은 비표준 형식이므로 폴더 이름을 변경했습니다.
OUTPUT_DIR = os.path.join(SUNRGBD_ROOT, 'polygons_name_output') 
CLASSES_FILE = os.path.join(SUNRGBD_ROOT, 'classes.txt') 

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
## 1. 클래스 이름 목록 수집 및 통합 ID 매핑 생성 (classes.txt 생성 목적)
# ----------------------------------------------------
def build_class_mapping():
    """
    NYU 데이터셋 전체에서 고유 클래스 이름을 수집하고, 
    classes.txt를 생성하는 용도로만 사용합니다. 
    """
    # 💡 검색 패턴: SUNRGBD/kv1/NYUdata/NYU000*/.../index.json
    search_pattern = os.path.join(SUNRGBD_ROOT, 'kv1', 'NYUdata', 'NYU000*', 'annotation', 'index.json')
    json_files = glob.glob(search_pattern, recursive=True)
    
    unique_classes = set()

    for idx_file in json_files:
        try:
            with open(idx_file, 'r') as f:
                data = json.load(f)
            
            top_level_objects = data.get("objects", [])
            
            for obj in top_level_objects:
                name = None
                if isinstance(obj, dict):
                    name = obj.get("name")
                elif isinstance(obj, str):
                    name = obj
                
                # 'name'이 유효하면 사용, 'null'이나 빈 문자열이면 건너뜀
                if name and isinstance(name, str) and name.strip():
                    unique_classes.add(name.strip())
                    
        except Exception:
            continue

    # classes.txt 파일 생성을 위한 정렬
    sorted_classes = sorted(list(unique_classes))
    
    # classes.txt 파일 생성 (YOLOv8 학습용)
    with open(CLASSES_FILE, 'w') as f:
        for name in sorted_classes:
            f.write(f"{name}\n")
    
    print(f"총 {len(sorted_classes)}개의 고유 클래스 이름 발견. {CLASSES_FILE} 저장 완료.")
    
    # 이 함수는 이름 -> 숫자 ID 매핑을 반환하지만, 
    # 아래 convert_json_to_name_seg 함수에서는 이 맵을 사용하지 않습니다.
    return {name: i for i, name in enumerate(sorted_classes)} 

# ----------------------------------------------------
## 2. JSON 파일을 클래스 이름 Segmentation TXT 파일로 변환 (비표준 형식)
# ----------------------------------------------------
def convert_json_to_name_seg(json_files):
    total_files = len(json_files)
    
    for file_index, idx_file in enumerate(json_files):
        print(f"--- ({file_index + 1}/{total_files}) 처리 중: {os.path.basename(idx_file)} ---")
        
        # 1. JSON 읽기
        try:
            with open(idx_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON 읽기 오류: {idx_file} - {e}")
            continue

        # 파일별 object ID (index) -> Class Name 매핑 생성
        file_object_map = {}
        top_level_objects = data.get("objects", [])
        
        for i, obj in enumerate(top_level_objects):
            name = None
            if isinstance(obj, dict):
                name = obj.get("name")
            elif isinstance(obj, str):
                name = obj
                
            if name and isinstance(name, str) and name.strip():
                # 인덱스(object ID)를 클래스 이름으로 매핑
                file_object_map[i] = name.strip()

        # 2. 이미지 파일 확인 및 크기 가져오기
        img_dir = os.path.dirname(os.path.dirname(idx_file))
        img_files = glob.glob(os.path.join(img_dir, 'image', '*.jpg'))
        
        if len(img_files) == 0:
            print(f"이미지 파일 없음: {img_dir}")
            continue
            
        img_path = img_files[0]
        
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            print(f"이미지 크기: {img_width}x{img_height}")
        except Exception as e:
            print(f"이미지 열기 오류: {img_path} - {e}")
            continue

        # 3. 텍스트 출력 파일 경로 설정
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        # 4. 비표준 레이블 생성 (클래스 이름 사용)
        name_annotations = []
        frames = data.get("frames", [])
        
        for frame in frames:
            polygons_to_process = frame.get("polygon", [frame])
            
            flat_polygons = []
            for p in polygons_to_process:
                if isinstance(p, list):
                    flat_polygons.extend(p)
                elif isinstance(p, dict):
                    flat_polygons.append(p)
            
            for p in flat_polygons:
                obj_id = p.get("object")
                
                # 4-1. Object ID로 클래스 이름 찾기 (문자열)
                class_name = file_object_map.get(obj_id)
                
                if class_name is None:
                    continue
                
                # --------------------------------------------------------
                # 2D 픽셀 좌표 가져오기 및 타입 처리 
                # --------------------------------------------------------
                x_data = p.get("x", [])
                y_data = p.get("y", [])

                x_list = [x_data] if not isinstance(x_data, list) else x_data
                y_list = [y_data] if not isinstance(y_data, list) else y_data
                
                if len(x_list) != len(y_list) or len(x_list) < 3:
                    continue
                # --------------------------------------------------------

                # 5. 좌표 정규화 및 포맷팅
                coords_str = []
                valid_coords = True
                for x, y in zip(x_list, y_list):
                    try:
                        x_norm = max(0.0, min(1.0, float(x) / img_width))
                        y_norm = max(0.0, min(1.0, float(y) / img_height))
                        coords_str.append(f"{x_norm:.6f} {y_norm:.6f}")
                    except (ValueError, TypeError):
                        valid_coords = False
                        break
                    
                if not valid_coords:
                    continue
                    
                # ⭐⭐⭐ 핵심 수정 부분: class_name 변수를 사용하여 문자열로 출력합니다. ⭐⭐⭐
                line = f"{class_name} {' '.join(coords_str)}" 
                name_annotations.append(line)

        # 6. TXT 파일 저장
        if name_annotations:
            try:
                with open(txt_path, 'w') as f_out:
                    f_out.write('\n'.join(name_annotations) + '\n')
                print(f"{txt_path} 저장 완료 (주석 {len(name_annotations)}개) ✅")
            except Exception as e:
                print(f"TXT 저장 오류: {txt_path} - {e}")
        else:
            print(f"주석이 없어 {os.path.basename(txt_path)}를 생성하지 않음.")

# ----------------------------------------------------
## 메인 실행
# ----------------------------------------------------
# classes.txt 생성을 위해 매핑 함수는 그대로 실행
build_class_mapping()

# 변환 대상 파일 검색
search_pattern = os.path.join(SUNRGBD_ROOT, '*', 'NYUdata', '**', 'annotation', 'index.json')
json_files_for_conversion = glob.glob(search_pattern, recursive=True)

print(f"\n총 {len(json_files_for_conversion)}개의 JSON 파일을 변환합니다.")
print(f"⚠️ 경고: '{OUTPUT_DIR}'에 저장되는 파일은 클래스 이름이 포함된 비표준 형식입니다! ⚠️")

# 변환 함수 실행
convert_json_to_name_seg(json_files_for_conversion)

print("\n--- 클래스 이름 변환 작업 완료 ---")