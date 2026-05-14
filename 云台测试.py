import paho.mqtt.client as mqtt
import json
import time

# =========================
# 配置
# =========================

REAL_IP = "192.168.43.123"

DEVICE_SN = "1581F7FVC252G00CZP6D"

USERNAME = "pilot_mqtt"

# =========================
# DJI MQTT 调试器
# =========================

class DjiDebugTracker:

    def __init__(self):

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        self.client.username_pw_set(USERNAME)

        self.client.on_connect = self.on_connect

        self.client.on_message = self.on_message

        self.msg_count = 0

    # =========================
    # MQTT连接
    # =========================
    def on_connect(
            self,
            client,
            userdata,
            flags,
            rc,
            properties=None
    ):

        if rc == 0:

            print("\n✅ MQTT连接成功")

            topics = [

                f"sys/product/{DEVICE_SN}/#",

                f"thing/product/{DEVICE_SN}/#"
            ]

            for t in topics:

                client.subscribe(t)

                print(f"📡 已订阅: {t}")

            print("\n🚀 开始监听 DJI 原始数据...\n")

        else:

            print(f"❌ MQTT连接失败: {rc}")

    # =========================
    # 收到消息
    # =========================
    def on_message(self, client, userdata, msg):

        self.msg_count += 1

        print("\n" + "=" * 80)

        print(f"📨 消息 #{self.msg_count}")

        print(f"\n📡 Topic:")
        print(msg.topic)

        try:

            payload_text = msg.payload.decode(
                "utf-8",
                errors="ignore"
            )

            print(f"\n📦 原始Payload:")
            print(payload_text)

            # JSON解析
            try:

                payload = json.loads(payload_text)

                print("\n🌲 JSON结构:")

                self.print_json_structure(payload)

                print("\n🔍 自动字段分析:")

                self.find_interesting_fields(payload)

            except Exception as e:

                print("\n⚠️ 非JSON数据")

        except Exception as e:

            print(f"\n❌ Payload解析失败: {e}")

        print("=" * 80)

    # =========================
    # 打印JSON结构
    # =========================
    def print_json_structure(
            self,
            data,
            prefix=""
    ):

        if isinstance(data, dict):

            for key, value in data.items():

                current = f"{prefix}.{key}" if prefix else key

                value_type = type(value).__name__

                # 简单值
                if isinstance(
                        value,
                        (str, int, float, bool)
                ):

                    print(
                        f"{current} "
                        f"[{value_type}] = {value}"
                    )

                else:

                    print(
                        f"{current} "
                        f"[{value_type}]"
                    )

                self.print_json_structure(
                    value,
                    current
                )

        elif isinstance(data, list):

            for i, item in enumerate(data):

                current = f"{prefix}[{i}]"

                self.print_json_structure(
                    item,
                    current
                )

    # =========================
    # 自动发现关键字段
    # =========================
    def find_interesting_fields(
            self,
            data,
            path=""
    ):

        keywords = [

            "lat",
            "lon",
            "lng",
            "latitude",
            "longitude",
            "gps",
            "sn",
            "aircraft",
            "drone",
            "uav",
            "osd",
            "height",
            "altitude",
            "gimbal",
            "attitude"
        ]

        if isinstance(data, dict):

            for key, value in data.items():

                current_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )

                lower_key = key.lower()

                # 命中关键字
                for kw in keywords:

                    if kw in lower_key:

                        print(
                            f"🎯 命中字段:"
                            f" {current_path}"
                        )

                        print(f"   值: {value}")

                        break

                self.find_interesting_fields(
                    value,
                    current_path
                )

        elif isinstance(data, list):

            for i, item in enumerate(data):

                self.find_interesting_fields(
                    item,
                    f"{path}[{i}]"
                )

    # =========================
    # 启动
    # =========================
    def run(self):

        print("🚀 正在连接 DJI MQTT...")

        self.client.connect(
            REAL_IP,
            1883,
            60
        )

        self.client.loop_forever()

# =========================
# 主程序
# =========================

if __name__ == "__main__":

    tracker = DjiDebugTracker()

    tracker.run()