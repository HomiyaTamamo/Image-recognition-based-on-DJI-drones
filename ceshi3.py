import cv2
import threading
import math
import json
import numpy as np

import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ==========================================
# DJI MQTT配置
# ==========================================

REAL_IP = "192.168.43.123"

DEVICE_SN = "1581F7FVC252G00CZP6D"

MQTT_PORT = 1883

USERNAME = "pilot_mqtt"

# ==========================================
# 相机参数
# ==========================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

HFOV = 84

# 针孔模型焦距
fx = IMAGE_WIDTH / (
        2 * math.tan(
    math.radians(HFOV / 2)
)
)

fy = fx

cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2

# 地球半径
EARTH_RADIUS = 6378137.0

# ==========================================
# 全局变量
# ==========================================

frame = None
result_frame = None

lock = threading.Lock()

running = True

# ==========================================
# 无人机状态
# ==========================================

drone_lat = 0.0
drone_lon = 0.0
drone_alt = 10.0

drone_yaw = 0.0
drone_pitch = -90.0
drone_roll = 0.0

mqtt_ok = False

# ==========================================
# DJI MQTT
# ==========================================

class DjiOSDTracker:

    def __init__(self):

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        self.client.username_pw_set(USERNAME)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    # ======================================
    # MQTT连接
    # ======================================

    def on_connect(
            self,
            client,
            userdata,
            flags,
            rc,
            properties=None
    ):

        global mqtt_ok

        if rc == 0:

            mqtt_ok = True

            topic = f"thing/product/{DEVICE_SN}/osd"

            client.subscribe(topic)

            print("✅ MQTT连接成功")
            print(f"📡 已订阅: {topic}")

        else:

            mqtt_ok = False

            print("❌ MQTT连接失败")

    # ======================================
    # OSD数据
    # ======================================

    def on_message(self, client, userdata, msg):

        global drone_lat
        global drone_lon
        global drone_alt

        global drone_yaw
        global drone_pitch
        global drone_roll

        try:

            payload = json.loads(
                msg.payload.decode(
                    "utf-8",
                    errors="ignore"
                )
            )

            data = payload.get("data", {})

            lat = data.get("latitude", 0)
            lon = data.get("longitude", 0)

            alt = data.get("height", 10)

            yaw = data.get("attitude_head", 0)
            pitch = data.get("attitude_pitch", -90)
            roll = data.get("attitude_roll", 0)

            if lat != 0 and lon != 0:

                drone_lat = lat
                drone_lon = lon

            drone_alt = abs(alt)

            drone_yaw = yaw
            drone_pitch = pitch
            drone_roll = roll

        except Exception as e:

            print("❌ OSD解析失败")
            print(e)

    # ======================================
    # 启动
    # ======================================

    def run(self):

        self.client.connect(
            REAL_IP,
            MQTT_PORT,
            60
        )

        self.client.loop_forever()

# ==========================================
# 旋转矩阵
# ==========================================

def rotation_matrix(yaw, pitch, roll):

    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    # Z轴旋转
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Y轴旋转
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    return Rz @ Ry @ Rx

# ==========================================
# 高精度定位算法
# ==========================================

def pixel_to_geo(u, v):

    global drone_lat
    global drone_lon
    global drone_alt

    global drone_yaw
    global drone_pitch
    global drone_roll

    # ======================================
    # 相机内参矩阵
    # ======================================

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # ======================================
    # 像素坐标
    # ======================================

    pixel = np.array([
        [u],
        [v],
        [1]
    ])

    # ======================================
    # 反投影
    # ======================================

    cam_ray = np.linalg.inv(K) @ pixel

    # 单位化
    cam_ray = cam_ray / np.linalg.norm(cam_ray)

    # ======================================
    # 姿态旋转
    # ======================================

    R = rotation_matrix(
        drone_yaw,
        drone_pitch,
        drone_roll
    )

    world_ray = R @ cam_ray

    # ======================================
    # 与地面求交
    # ======================================

    if world_ray[2][0] >= 0:

        return drone_lat, drone_lon

    t = drone_alt / abs(world_ray[2][0])

    dx = world_ray[0][0] * t
    dy = world_ray[1][0] * t

    # ======================================
    # 米 -> 经纬度
    # ======================================

    target_lat = (
            drone_lat
            + (dy / EARTH_RADIUS)
            * (180 / math.pi)
    )

    target_lon = (
            drone_lon
            + (
                    dx /
                    (
                            EARTH_RADIUS
                            * math.cos(
                        math.radians(drone_lat)
                    )
                    )
            )
            * (180 / math.pi)
    )

    return target_lat, target_lon

# ==========================================
# 绘制OSD信息
# ==========================================

def draw_osd(img):

    overlay = img.copy()

    # 半透明背景
    cv2.rectangle(
        overlay,
        (10, 10),
        (420, 220),
        (0, 0, 0),
        -1
    )

    alpha = 0.5

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img
    )

    info_list = [

        f"Drone Latitude : {drone_lat:.6f}",
        f"Drone Longitude: {drone_lon:.6f}",
        f"Drone Altitude : {drone_alt:.2f} m",

        f"Yaw   : {drone_yaw:.2f}",
        f"Pitch : {drone_pitch:.2f}",
        f"Roll  : {drone_roll:.2f}",

        f"MQTT  : {'CONNECTED' if mqtt_ok else 'DISCONNECTED'}"
    ]

    y = 40

    for text in info_list:

        cv2.putText(
            img,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += 28

# ==========================================
# 视频采集线程
# ==========================================

def capture_thread(cap):

    global frame
    global running

    while running:

        ret, img = cap.read()

        if not ret:
            continue

        with lock:

            frame = img.copy()

# ==========================================
# YOLO线程
# ==========================================

def detect_thread(model):

    global frame
    global result_frame
    global running

    while running:

        if frame is None:
            continue

        with lock:

            img = frame.copy()

        # YOLO检测
        results = model(
            img,
            conf=0.4,
            imgsz=640
        )

        annotated = img.copy()

        # ======================================
        # 绘制无人机OSD
        # ======================================

        draw_osd(annotated)

        # ======================================
        # 目标检测
        # ======================================

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # 中心点
                cx_box = int((x1 + x2) / 2)
                cy_box = int((y1 + y2) / 2)

                # ==================================
                # 高精度定位
                # ==================================

                target_lat, target_lon = pixel_to_geo(
                    cx_box,
                    cy_box
                )

                # 框
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                # 中心点
                cv2.circle(
                    annotated,
                    (cx_box, cy_box),
                    5,
                    (0,0,255),
                    -1
                )

                # 标签
                label = (
                    f"{model.names[cls]} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

                # 经纬度
                geo_text = (
                    f"{target_lat:.6f}, "
                    f"{target_lon:.6f}"
                )

                cv2.putText(
                    annotated,
                    geo_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2
                )

                # 控制台输出
                print("\n🎯 检测到目标")

                print(
                    f"目标GPS: "
                    f"{target_lat:.6f}, "
                    f"{target_lon:.6f}"
                )

        with lock:

            result_frame = annotated

# ==========================================
# 主函数
# ==========================================

def main():

    global running

    # MQTT线程
    mqtt_tracker = DjiOSDTracker()

    mqtt_thread = threading.Thread(
        target=mqtt_tracker.run,
        daemon=True
    )

    mqtt_thread.start()

    # HDMI视频
    cap = cv2.VideoCapture(
        1,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("❌ 无法打开视频设备")

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        IMAGE_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        IMAGE_HEIGHT
    )

    # YOLO模型
    model = YOLO("yolov8s.pt")

    # 启动线程
    t1 = threading.Thread(
        target=capture_thread,
        args=(cap,)
    )

    t2 = threading.Thread(
        target=detect_thread,
        args=(model,)
    )

    t1.start()
    t2.start()

    print("\n✅ 无人机实时定位系统启动")
    print("ESC退出\n")

    # 主循环
    while True:

        if result_frame is not None:

            cv2.imshow(
                "Drone Real-Time Detection",
                result_frame
            )

        key = cv2.waitKey(1)

        if key == 27:

            running = False

            break

    # 释放
    t1.join()
    t2.join()

    cap.release()

    cv2.destroyAllWindows()

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":

    main()