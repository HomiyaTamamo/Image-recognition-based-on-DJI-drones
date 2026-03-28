import cv2
from ultralytics import YOLO


def detect_people_yolo(input_video, output_video, display=False):

    # 加载YOLO模型（推荐 nano 或 small）
    model = YOLO("yolov8n.pt")   # 自动下载模型

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("无法打开视频")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # YOLO检测
        results = model(frame, conf=0.4)

        person_count = 0

        for r in results:
            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])

                # COCO类别0是person
                if cls == 0:

                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    conf = float(box.conf[0])

                    # 画框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    label = f"person {conf:.2f}"

                    cv2.putText(frame, label,
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,0),
                                1)

        # 显示人数
        cv2.putText(frame,
                    f"People: {person_count}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        out.write(frame)

        if display:
            cv2.imshow("YOLO Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()
    out.release()

    if display:
        cv2.destroyAllWindows()

    print("处理完成，输出视频：", output_video)


if __name__ == "__main__":

    input_video = "ceshi1.mp4"
    output_video = "ceshi1pro.mp4"

    detect_people_yolo(input_video, output_video, display=False)