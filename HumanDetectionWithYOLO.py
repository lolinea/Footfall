import cv2
from ultralytics import YOLO

SOURCE = "sample.mp4"
WEIGHTS = "yolov8n.pt"
CONF = 0.2 #tried as 0.1, but trashcan or random stuffs get detected as people
IMGSZ = 1280
SHOW_WIDTH = 1280

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, classes=[0], conf=CONF, imgsz=IMGSZ, verbose=False)
    boxes = results[0].boxes

    count = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.putText(frame, f"People: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    h, w = frame.shape[:2]
    if w > SHOW_WIDTH:
        frame = cv2.resize(frame, (SHOW_WIDTH, int(h * SHOW_WIDTH / w)))

    cv2.imshow("PersonDetection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()