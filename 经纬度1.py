import paho.mqtt.client as mqtt
import json
import time

# --- 核心配置 ---
# 请确保使用 ipconfig 查到的 192.168.43.xxx
REAL_IP = "192.168.43.123"
DEVICE_SN = "9N9CN2G00129QX"
# 这玩意是这个用不上的无人机的SN
DRONE_SN = "1581F7FVC252G00CZP6D"


class DjiFullDataTracker:
    def __init__(self):
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set("pilot_mqtt")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"✅ 链路已就绪，正在实时提取硬件数据...")
            # 订阅全部相关 Topic
            client.subscribe(f"sys/product/{DEVICE_SN}/#")
            client.subscribe(f"thing/product/{DEVICE_SN}/#")
        else:
            print(f"❌ 连接失败，状态码: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic

            # 1. 协议修复逻辑 (保持握手)
            if "status" in topic and payload.get("method") == "update_topo":
                reply = {
                    "tid": payload.get("tid"),
                    "bid": payload.get("bid"),
                    "method": "update_topo",
                    "timestamp": int(time.time() * 1000),
                    "data": {"result": 0}
                }
                client.publish(f"{topic}_reply", json.dumps(reply))

            # 2. 核心数据解析 (OSD)
            if "osd" in topic:
                data = payload.get("data", {})

                # 提取坐标
                lon = data.get("longitude", 0)
                lat = data.get("latitude", 0)
                alt = data.get("height", 0)

                # 提取云台姿态 (Gimbal)
                gimbal = data.get("gimbal", {})
                g_pitch = gimbal.get("pitch", "N/A")
                g_roll = gimbal.get("roll", "N/A")
                g_yaw = gimbal.get("yaw", "N/A")

                # 提取无人机姿态 (Aircraft Attitude)
                attitude = data.get("attitude", {})
                a_pitch = attitude.get("pitch", "N/A")

                print("\n" + "=" * 40)
                print(f"📡 实时遥测数据 | Topic: .../osd")
                print(f"📍 位置: 经度 {lon}, 纬度 {lat}, 高度 {alt}m")
                print(f"🎥 云台姿态: Pitch(俯仰) {g_pitch}°, Yaw(偏航) {g_yaw}°")
                print(f"🛸 飞机姿态: Pitch {a_pitch}°")
                print("=" * 40)

        except Exception as e:
            pass

    def run(self):
        self.client.connect(REAL_IP, 1883, 60)
        self.client.loop_forever()


if __name__ == "__main__":
    tracker = DjiFullDataTracker()
    tracker.run()