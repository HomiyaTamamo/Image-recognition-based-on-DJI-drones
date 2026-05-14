import paho.mqtt.client as mqtt
import json
import time

# ==========================================
# 配置
# ==========================================

REAL_IP = "192.168.43.123"

# 飞机 SN
DEVICE_SN = "1581F7FVC252G00CZP6D"

MQTT_PORT = 1883

USERNAME = "pilot_mqtt"

# ==========================================
# DJI OSD 监听器
# ==========================================

class DjiOSDTracker:

    def __init__(self):

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        self.client.username_pw_set(USERNAME)

        self.client.on_connect = self.on_connect

        self.client.on_message = self.on_message

        self.last_print_time = 0

        self.msg_count = 0

    # ==========================================
    # MQTT连接
    # ==========================================
    def on_connect(
            self,
            client,
            userdata,
            flags,
            rc,
            properties=None
    ):

        if rc == 0:

            print("\n" + "=" * 60)

            print("✅ MQTT连接成功")

            print(f"📡 Broker: {REAL_IP}:{MQTT_PORT}")

            # 只订阅 OSD
            topic = f"thing/product/{DEVICE_SN}/osd"

            client.subscribe(topic)

            print(f"✅ 已订阅OSD: {topic}")

            print("=" * 60)

        else:

            print(f"❌ MQTT连接失败: {rc}")

    # ==========================================
    # 收到OSD
    # ==========================================
    def on_message(self, client, userdata, msg):

        self.msg_count += 1

        topic = msg.topic

        try:

            payload_text = msg.payload.decode(
                "utf-8",
                errors="ignore"
            )

            payload = json.loads(payload_text)

            data = payload.get("data", {})

            # ======================================
            # GPS状态
            # ======================================

            gps_state = data.get(
                "position_state",
                {}
            )

            gps_num = gps_state.get(
                "gps_number",
                0
            )

            is_fixed = gps_state.get(
                "is_fixed",
                0
            )

            quality = gps_state.get(
                "quality",
                0
            )

            # ======================================
            # 经纬度
            # ======================================

            latitude = data.get(
                "latitude",
                0
            )

            longitude = data.get(
                "longitude",
                0
            )

            # ======================================
            # 飞机姿态
            # ======================================

            pitch = data.get(
                "attitude_pitch",
                0
            )

            roll = data.get(
                "attitude_roll",
                0
            )

            yaw = data.get(
                "attitude_head",
                0
            )

            # ======================================
            # 高度
            # ======================================

            height = data.get(
                "height",
                0
            )

            # ======================================
            # 电池
            # ======================================

            battery = data.get(
                "battery",
                {}
            )

            battery_percent = battery.get(
                "capacity_percent",
                0
            )

            # ======================================
            # 限制打印频率
            # ======================================

            now = time.time()

            if now - self.last_print_time < 1:
                return

            self.last_print_time = now

            # ======================================
            # 输出
            # ======================================

            print("\n" + "=" * 70)

            print(f"📨 OSD数据包 #{self.msg_count}")

            print(f"\n📡 Topic:")
            print(topic)

            print("\n🛰 GPS状态")

            print(f"搜星数量: {gps_num}")

            print(f"GPS定位: {'✅' if is_fixed else '❌'}")

            print(f"GPS质量: {quality}")

            # ======================================
            # 经纬度判断
            # ======================================

            if (
                    is_fixed == 1
                    and latitude != 0
                    and longitude != 0
            ):

                print("\n✅ 已获取真实GPS坐标")

                print(f"📍 纬度: {latitude}")

                print(f"📍 经度: {longitude}")

            else:

                print("\n⌛ GPS尚未定位")

                print(f"当前纬度: {latitude}")

                print(f"当前经度: {longitude}")

            # ======================================
            # 飞机状态
            # ======================================

            print("\n🛸 飞机状态")

            print(f"高度: {height} m")

            print(f"Pitch: {pitch}")

            print(f"Roll : {roll}")

            print(f"Yaw  : {yaw}")

            # ======================================
            # 电池
            # ======================================

            print("\n🔋 电池")

            print(f"{battery_percent}%")

            print("=" * 70)

        except Exception as e:

            print("\n❌ OSD解析失败")

            print(e)

    # ==========================================
    # 启动
    # ==========================================
    def run(self):

        print("🚀 正在连接 DJI MQTT...")

        self.client.connect(
            REAL_IP,
            MQTT_PORT,
            60
        )

        self.client.loop_forever()

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":

    tracker = DjiOSDTracker()

    tracker.run()