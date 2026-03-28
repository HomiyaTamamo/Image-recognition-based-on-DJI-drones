import cv2
import threading
from ultralytics import YOLO

# ======================
# 全局变量（共享帧）
# ======================
frame = None
result_frame = None
lock = threading.Lock()

running = True


# ======================
# 线程1：视频采集（HDMI）
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
# 线程2：YOLO检测
# ======================
def detect_thread(model):
    global frame, result_frame, running

    while running:

        if frame is None:
            continue

        with lock:
            img = frame.copy()

        # YOLO检测（降低分辨率提高速度）
        results = model(img, conf=0.3, imgsz=640)

        annotated = results[0].plot()

        with lock:
            result_frame = annotated


# ======================
# 主程序
# ======================
def main():

    global running

    # 打开HDMI设备（根据你的设备调整）
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ 无法打开视频设备")
        return

    # 降低分辨率提高实时性
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 加载YOLO模型
    model = YOLO("yolov8s.pt")

    # 启动线程
    t1 = threading.Thread(target=capture_thread, args=(cap,))
    t2 = threading.Thread(target=detect_thread, args=(model,))

    t1.start()
    t2.start()

    print("✅ 实时检测启动，按 ESC 退出")

    while True:

        if result_frame is not None:
            cv2.imshow("Drone YOLO Detection", result_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            running = False
            break

    t1.join()
    t2.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()