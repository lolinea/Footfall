import os
import cv2
from ultralytics import YOLO, YOLOWorld

SOURCE = 'sample.mp4'
SHOW_WIDTH = 1280
SAVE_DIR = 'labeling'

PERSON_WEIGHTS = 'yolo26s.pt'
PERSON_CONF = 0.2
PERSON_IMGSZ = 1280

BADGE_WEIGHTS = 'yolov8s-worldv2.pt'
BADGE_PROMPTS = ['name tag', 'id card']
BADGE_CONF = 0.10
BADGE_IMGSZ = 640
CROP_PAD = 0.10

personModel = YOLO(PERSON_WEIGHTS)
badgeModel = YOLOWorld(BADGE_WEIGHTS)
badgeModel.set_classes(BADGE_PROMPTS)

os.makedirs(SAVE_DIR, exist_ok=True)
saved = 0

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print('Cannot open source')
    exit()

while True:
    ready, frame = cap.read()
    if not ready:
        break

    raw = frame.copy()
    HEIGHT, WIDTH = frame.shape[:2]
    badgeBoxes = []

    people = personModel.predict(
        source=raw, classes=[0], conf=PERSON_CONF, imgsz=PERSON_IMGSZ, verbose=False
    )[0].boxes

    for person in people:
        x1, y1, x2, y2 = map(int, person.xyxy[0].tolist())

        paddingX = int((x2 - x1) * CROP_PAD)
        paddingY = int((y2 - y1) * CROP_PAD)
        centerX1 = max(0, x1 - paddingX)
        centerY1 = max(0, y1 - paddingY)
        centerX2 = min(WIDTH, x2 + paddingX)
        centerY2 = min(HEIGHT, y2 + paddingY)
        crop = raw[centerY1:centerY2, centerX1:centerX2]

        if crop.size == 0:
            continue

        badges = badgeModel.predict(
            source=crop, conf=BADGE_CONF, imgsz=BADGE_IMGSZ, verbose=False
        )[0].boxes

        if len(badges) > 0:
            best = badges[int(badges.conf.argmax())]
            bx1, by1, bx2, by2 = map(int, best.xyxy[0].tolist())
            bconf = float(best.conf[0])

            fx1, fy1 = bx1 + centerX1, by1 + centerY1
            fx2, fy2 = bx2 + centerX1, by2 + centerY1
            badgeBoxes.append((fx1, fy1, fx2, fy2))

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f'badge? {bconf:.2f}',
                (fx1, max(20, fy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            personColor = (0, 255, 0)
        else:
            personColor = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), personColor, 2)

    cv2.putText(
        frame,
        f'People: {len(people)}  Badge candidates: {len(badgeBoxes)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    if WIDTH > SHOW_WIDTH:
        frame = cv2.resize(frame, (SHOW_WIDTH, int(HEIGHT * SHOW_WIDTH / WIDTH)))

    cv2.imshow('BadgeDetection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    if key == ord('s'):
        name = f'frame_{saved:05d}'
        cv2.imwrite(os.path.join(SAVE_DIR, name + '.jpg'), raw)
        with open(os.path.join(SAVE_DIR, name + '.txt'), 'w') as f:
            for fx1, fy1, fx2, fy2 in badgeBoxes:
                xc = (fx1 + fx2) / 2 / WIDTH
                yc = (fy1 + fy2) / 2 / HEIGHT
                bw = (fx2 - fx1) / WIDTH
                bh = (fy2 - fy1) / HEIGHT
                f.write(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
        saved += 1
        print('saved', name)

cap.release()
cv2.destroyAllWindows()