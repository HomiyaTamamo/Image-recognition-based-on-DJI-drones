import cv2
import time

def preview_hdmi_debug(device_id=1):

    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ HDMI设备打开失败")
        return

    # 设置分辨率（可调）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ HDMI调试模式启动")
    print("ESC：退出 | S：截图")

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 计算FPS
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        h, w = frame.shape[:2]

        # 显示信息
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Resolution: {w}x{h}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("HDMI Debug", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 已保存 {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    preview_hdmi_debug(1)