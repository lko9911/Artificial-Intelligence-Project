import cv2
from pygrabber.dshow_graph import FilterGraph
from prototype_alarm_new import *

def get_camera_list():
    graph = FilterGraph()
    # 시스템에 연결된 비디오 입력 장치 이름들을 리스트로 가져옵니다.
    devices = graph.get_input_devices()
    
    for index, name in enumerate(devices):
        if ("OBS" in name):
            return index

cap = cv2.VideoCapture(get_camera_list())

if not cap.isOpened():
    print("failed.")
    exit()

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

print("카메라 연결 성공! 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Streaming...', detecting(frame))

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()