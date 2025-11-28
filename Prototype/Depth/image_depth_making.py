import os
import glob
import json

# SUNRGBD 루트 폴더
SUNRGBD_ROOT = "SUNRGBD"

# index.json 파일을 재귀적으로 찾기
json_files = glob.glob(os.path.join(SUNRGBD_ROOT, "**", "annotation", "index.json"), recursive=True)
print(f"총 {len(json_files)}개의 index.json 파일 발견")

for json_path in json_files:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # txt 파일 경로: index.json이 있는 폴더 기준으로 동일 폴더에 저장
    txt_path = os.path.join(os.path.dirname(json_path), "polygons.txt")
    
    with open(txt_path, "w") as f_out:
        frames = data.get("frames", [])
        for frame in frames:
            polygons = frame.get("polygon", [])
            for poly in polygons:
                obj_id = poly.get("object", -1)
                x_list = poly.get("x", [])
                y_list = poly.get("y", [])
                
                if not x_list or not y_list:
                    continue
                
                x_str = ",".join(map(str, x_list))
                y_str = ",".join(map(str, y_list))
                
                f_out.write(f"{obj_id} {x_str} {y_str}\n")
    
    print(f"{txt_path} 저장 완료")
