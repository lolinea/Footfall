import json
import cv2

video = "sample.mp4"
outFile = "anchorsList.json"

cap = cv2.VideoCapture(video)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

anchors = []
frame_no = 0
loaded_no = -1
image = None


def load_frame(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, img = cap.read()
    return img


def on_click(event, x, y, flags, param):
    global anchors
    if event == cv2.EVENT_LBUTTONDOWN:
        anchors = [a for a in anchors if a["frame"] != frame_no]
        anchors.append({"frame": frame_no, "x": x, "y": y})
        anchors.sort(key=lambda a: a["frame"])
        print("added anchor:", frame_no, x, y)


cv2.namedWindow("pick anchors")
cv2.setMouseCallback("pick anchors", on_click)

while True:
    if frame_no != loaded_no:
        image = load_frame(frame_no)
        loaded_no = frame_no
    if image is None:
        frame_no = max(0, frame_no - 1)
        loaded_no = -1
        continue

    shown = image.copy()
    cv2.putText(shown, f"frame {frame_no}/{n_frames - 1}", (10, 25), 0, 0.7, (255, 255, 255), 2)
    for a in anchors:
        if a["frame"] == frame_no:
            cv2.circle(shown, (a["x"], a["y"]), 6, (0, 0, 255), -1)
    cv2.imshow("pick anchors", shown)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("d"):
        frame_no = min(frame_no + 1, n_frames - 1)
    elif key == ord("a"):
        frame_no = max(frame_no - 1, 0)
    elif key == ord("w"):
        frame_no = min(frame_no + 25, n_frames - 1)
    elif key == ord("q"):
        frame_no = max(frame_no - 25, 0)
    elif key == ord("k"):
        break

json.dump(anchors, open(outFile, "w"), indent=2)
cv2.destroyAllWindows()