import cv2
import threading
import math
from ultralytics import YOLO

# ======================
# 基准参数（可改）
# ======================
BASE_LAT = 30.0
BASE_LON = 114.0
ALTITUDE = 10.0
FOV = 84

# ======================
# 全局变量
# ======================
frame = None
result_frame = None
lock = threading.Lock()
running = True


# ======================
# 像素 → 经纬度
# ======================
def pixel_to_geo(x, y, w, h):

    ground_w = 2 * ALTITUDE * math.tan(math.radians(FOV/2))

    dx = (x/w - 0.5) * ground_w
    dy = (y/h - 0.5) * ground_w

    lat = BASE_LAT + dy / 111111
    lon = BASE_LON + dx / (111111 * math.cos(math.radians(BASE_LAT)))

    return lat, lon


# ======================
# 视频采集线程
# ======================
def capture_thread(cap):
    global frame, running

    while running:
        ret, img = cap.read()
        if not ret:
            continue

        with lock:
            frame = img.copy()


# ======================
# YOLO + 定位线程
# ======================
def detect_thread(model):
    global frame, result_frame, running

    while running:

        if frame is None:
            continue

        with lock:
            img = frame.copy()

        h, w = img.shape[:2]

        results = model(img, conf=0.3, imgsz=640)

        for r in results:
            for box in r.boxes:

                if int(box.cls[0]) == 0:  # person

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 中心点
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 经纬度计算
                    lat, lon = pixel_to_geo(cx, cy, w, h)

                    # 画框
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

                    # 中心点
                    cv2.circle(img, (cx,cy), 4, (0,0,255), -1)

                    # 显示经纬度
                    text = f"{lat:.6f},{lon:.6f}"

                    cv2.putText(img, text,
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0,255,0),
                                1)

        with lock:
            result_frame = img


# ======================
# 主函数
# ======================
def main():
    global running

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ 无法打开HDMI设备")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model = YOLO("yolov8s.pt")

    t1 = threading.Thread(target=capture_thread, args=(cap,))
    t2 = threading.Thread(target=detect_thread, args=(model,))

    t1.start()
    t2.start()

    print("✅ 实时定位系统启动（ESC退出）")

    while True:

        if result_frame is not None:
            cv2.imshow("Drone Geo Detection", result_frame)

        if cv2.waitKey(1) == 27:
            running = False
            break

    t1.join()
    t2.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()