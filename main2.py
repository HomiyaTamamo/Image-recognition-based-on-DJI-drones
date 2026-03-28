import cv2
import math
from ultralytics import YOLO


# 摄像机参数
CAMERA_LAT = 30.0
CAMERA_LON = 114.0
CAMERA_HEIGHT = 10.0
FOV = 90


def pixel_to_geo(x, y, width, height):
    """像素坐标转经纬度"""

    ground_width = 2 * CAMERA_HEIGHT * math.tan(math.radians(FOV / 2))

    dx = (x / width - 0.5) * ground_width
    dy = (y / height - 0.5) * ground_width

    lat_offset = dy / 111111
    lon_offset = dx / (111111 * math.cos(math.radians(CAMERA_LAT)))

    lat = CAMERA_LAT + lat_offset
    lon = CAMERA_LON + lon_offset

    return lat, lon


def detect_and_locate(video_in, video_out):

    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_in)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)

        people = []

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                if cls == 0:  # person

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    lat, lon = pixel_to_geo(cx, cy, width, height)

                    people.append((lat, lon))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

                    label = f"{lat:.6f},{lon:.6f}"

                    cv2.putText(frame,
                                label,
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0,255,0),
                                1)

        cv2.putText(frame,
                    f"People: {len(people)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        out.write(frame)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect_and_locate("ceshi1.mp4", "ceshi1_located.mp4")