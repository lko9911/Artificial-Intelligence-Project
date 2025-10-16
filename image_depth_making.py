import os
import shutil
import glob

# 1. SUNRGBD 데이터셋의 루트 폴더를 지정하세요.
SUNRGBD_ROOT = 'SUNRGBD'

# 2. 이미지를 모을 새 폴더를 지정하세요.
TARGET_DIR = os.path.join(SUNRGBD_ROOT, 'images')

# 타겟 디렉토리가 없으면 생성
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# 3. 'image' 폴더 안에 있는 모든 jpg 파일을 재귀적으로 검색
# **/* : 모든 하위 폴더 (재귀적 검색)
# */image : 'image'라는 이름의 폴더
# /*.jpg : 그 안의 모든 .jpg 파일
search_pattern = os.path.join(SUNRGBD_ROOT, '**', 'image', '*.jpg')
image_files = glob.glob(search_pattern, recursive=True)

print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

# 4. 파일을 타겟 폴더로 복사
for img_path in image_files:
    # 파일 이름만 추출 (예: img_0063.jpg)
    file_name = os.path.basename(img_path)
    # 중복 방지를 위해 경로 정보(상위 폴더명 등)를 이름에 추가할 수도 있음.
    # 예: "kv1_b3dodata_img_0063.jpg"

    # 복사 (이동을 원하시면 shutil.move 사용)
    try:
        shutil.copy2(img_path, os.path.join(TARGET_DIR, file_name))
    except Exception as e:
        print(f"파일 복사 중 오류 발생: {img_path} - {e}")

print(f"모든 이미지 파일이 {TARGET_DIR}에 복사되었습니다.")