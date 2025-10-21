import os
import glob

# --- 1. 사용자 정의 클래스 목록 및 ID 매핑 설정 ---
# 클래스 순서가 ID (0, 1, 2, ...) 순서가 됩니다.
USER_DEFINED_CLASSES = [
    "airduct", "airvent", "bag", "ball", "bar", "basket", "book", "bottle", "bowl", "box",
    "cabinet", "camera", "ceiling", "chair", "clock", "computer", "cone", "corkboard", "counter",
    "cup", "desk", "dishwasher", "door", "doorknob", "faucet", "fireextinguisher", "floor",
    "garbagebin", "greenscreen", "holepuncher", "keyboard", "ladder", "laptop", "light",
    "magnet", "manillaenvelope", "mantel", "microwave", "monitor", "motioncamera", "paper",
    "papertoweldispenser", "picture", "pipe", "pot", "projectorscreen", "refridgerator",
    "scissor", "shelves", "sink", "speaker", "stackedchairs", "stand", "stoveburner",
    "styrofoamobject", "table", "tapedispenser", "telephone", "telephonecord", "tracklight",
    "unknown", "wall", "whiteboard", "window"
]

# 클래스 이름 -> ID 매핑 생성
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(USER_DEFINED_CLASSES)}

# --- 2. 설정 (CONFIG) ---
SUNRGBD_ROOT = 'SUNRGBD'
# 재라벨링 대상 폴더
TARGET_DIR = os.path.join(SUNRGBD_ROOT, 'polygons_name_output')
# 새로운 클래스 ID 순서에 맞춰 classes.txt 파일도 생성
CLASSES_FILE_PATH = os.path.join(SUNRGBD_ROOT, 'final_yolo_classes.txt') 

# ----------------------------------------------------
## 3. 클래스 파일 생성
# ----------------------------------------------------
try:
    with open(CLASSES_FILE_PATH, 'w') as f:
        for name in USER_DEFINED_CLASSES:
            f.write(f"{name}\n")
    print(f"새 클래스 목록 파일 ({CLASSES_FILE_PATH}) 저장 완료. (총 {len(USER_DEFINED_CLASSES)}개 클래스)")
except Exception as e:
    print(f"클래스 파일 저장 중 오류 발생: {e}")

# ----------------------------------------------------
## 4. 재라벨링 함수 실행
# ----------------------------------------------------
def relabel_txt_files(target_dir, class_map):
    txt_files = glob.glob(os.path.join(target_dir, '*.txt'))
    total_files = len(txt_files)
    processed_count = 0
    
    print(f"\n총 {total_files}개의 TXT 파일을 사용자 정의 순서로 재라벨링합니다. 🔄")

    for file_index, txt_path in enumerate(txt_files):
        print(f"--- ({file_index + 1}/{total_files}) 처리 중: {os.path.basename(txt_path)} ---")
        
        new_lines = []
        changed_count = 0
        
        # 파일 읽기
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"파일 읽기 오류: {txt_path} - {e}")
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue
                
            old_class_name = parts[0] # 현재 파일의 클래스 이름(문자열)
            coordinates = parts[1:]
            
            # 사용자 정의 맵에서 새로운 ID 찾기
            new_id = class_map.get(old_class_name)

            if new_id is not None:
                # 새로운 ID로 라인 구성 (YOLO 표준 형식: ID + 좌표)
                new_line = f"{new_id} {' '.join(coordinates)}"
                new_lines.append(new_line)
                changed_count += 1
            else:
                # 맵에 없는 클래스는 오류를 표시하고 건너뜀
                print(f"경고: 사용자 정의 목록에 없는 클래스 이름 '{old_class_name}'이 발견되어 해당 라인을 건너뜁니다.")
        
        # 원본 파일 덮어쓰기
        if changed_count > 0:
            try:
                with open(txt_path, 'w') as f_out:
                    f_out.write('\n'.join(new_lines) + '\n')
                print(f"{os.path.basename(txt_path)} 재라벨링 완료. (클래스 ID {changed_count}개로 변경) ✅")
                processed_count += 1
            except Exception as e:
                print(f"파일 쓰기 오류: {txt_path} - {e}")
        else:
            print(f"참고: {os.path.basename(txt_path)}에서 변환된 유효한 라인이 없습니다.")


    print(f"\n--- 재라벨링 작업 완료. 총 {processed_count}개의 파일이 처리되었습니다. ---")

# --- 메인 실행 ---
relabel_txt_files(TARGET_DIR, CLASS_NAME_TO_ID)

print("\n**처리 결과:**")
print(f"'{TARGET_DIR}' 내의 모든 TXT 파일이 **사용자 정의 순서에 따른 숫자 ID**로 덮어쓰기되었습니다.")
print(f"새로운 클래스 ID 순서는 '{CLASSES_FILE_PATH}' 파일에서 확인할 수 있습니다.")