from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("model/best.pt")

# Define path to the image file - 구조 확인하고 넣기
source = "SUNRGBD/kv1/NYUdata/NYU0001/image/NYU0001.jpg"

model.predict(source, save=True, conf=0.5, show=True)