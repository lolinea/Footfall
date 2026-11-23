import json
import csv
import cv2
import numpy as np

WARMUP_FRAMES = 30       
MIN_BLOB_AREA = 1500     
MAX_BLOB_AREA = 60000   
MAX_STEP = 60            
AREA_RATIO = (0.4, 2.5) 
HIST_MAX = 0.55          
STUCK_WINDOW = 10        
STUCK_MIN_MOVE = 4     
MIN_SEGMENT = 5 

VIDEO = 'sample.mp4'
ANCHORS = 'anchorsList.json'


def colorHist(hsv, regionMask):
    #Checks whether the person in anchors has the same color
    inner = cv2.erode(regionMask, np.ones((5, 5), np.uint8), iterations=2)

    if inner.sum() < 100:
        inner = regionMask

    h = cv2.calcHist([hsv], [0, 1, 2], inner, [8, 4, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1).flatten().astype(np.float32)

def colorDistance(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

def findBlobs(videoPath):
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows = False)
    kernel = np.ones((5, 5), np.uint8)

    allBlobs = []
    i = 0

    while True: 
        ready, frame = cap.read()

        if not ready: 
            break

        mask = bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=3)
        blobs = []

        if (i >= WARMUP_FRAMES) and ((mask > 0).mean() < 0.3):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)

            for j in range(1, n):
                x, y, w, h, area = stats[j]

                if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                    region = (labels == j).astype(np.uint8) * 255
                    blobs.append({
                        'centerX' : float(centers[j][0]), 
                        'centerY' : float(centers[j][1]), 
                        'box' : (int(x), int(y), int(w), int(h)), 
                        'area' : int(area), 
                        'hist' : colorHist(hsv, region)
                    })

        allBlobs.append(blobs)
        i += 1

    cap.release()
    return fps, WIDTH, HEIGHT, allBlobs

def blobAtPoint(blobs, x, y):
    best, bestDistance = None, 1e9

    for b in blobs:
        blobX, blobY, blobWidth, blobHeight = b['box']
        inside = (blobX <= x <= blobX + blobWidth) and (blobY <= y <= blobY + blobHeight)

        distance = np.hypot(b['centerX'] - x, b['centerY'] - y) - (1000 if inside else 0)

        if distance < bestDistance:
            best, bestDistance = b, distance


    if best is None or bestDistance > 80:
        return None

    return best


def follow(allBlobs, startFrame, startBlob, step):
    refHist = startBlob['hist']
    centerX, centerY, area = startBlob['centerX'], startBlob['centerY'], startBlob['area']

    velocityX = velocityY = 0.0
    history = [(centerX, centerY)]
    found = {}
    frame = startFrame + step

    while 0 <= frame < len(allBlobs):
        personX, personY = centerX + velocityX, centerY + velocityY
        best, bestCost, bestHistDist = None, 1e9, 1.0

        for blob in allBlobs[frame]:
            dist = np.hypot(blob['centerX'] - personX, blob['centerY'] - personY)
            ratio = blob['area'] / area
            histDist = colorDistance(refHist, blob['hist'])

            if dist > MAX_STEP or not (AREA_RATIO[0] <= ratio <= AREA_RATIO[1]) or histDist > HIST_MAX:
                continue

            cost = dist + 100 * histDist
            
            if cost < bestCost:
                best, bestCost, bestHistDist = blob, cost, histDist

        if best is None: 
            break

        velocityX = 0.5 * velocityX + 0.5 * (best['centerX'] - centerX)
        velocityY = 0.5 * velocityY + 0.5 * (best['centerY'] - centerY)

        centerX, centerY, area = best['centerX'], best['centerY'], best['area']
        history.append((centerX, centerY))

        if len(history) > STUCK_WINDOW:
            previousX, previousY = history[-STUCK_WINDOW - 1]

            if np.hypot(centerX - previousX, centerY - previousY) < STUCK_MIN_MOVE:
                for stuck in range(STUCK_WINDOW):
                    found.pop(frame - step * stuck, None)
                break


        found[frame] = (best, bestHistDist)
        frame += step

    return found
                
def main():
    anchors = json.load(open(ANCHORS))
    fps, WIDTH, HEIGHT, allBlobs = findBlobs(VIDEO)
    nFrames = len(allBlobs)

    present = {}

    for anchor in anchors:
        frame = anchor['frame']
        blob = blobAtPoint(allBlobs[frame], anchor['x'], anchor['y'])

        if blob is None:
            print("no blob")
            continue

        tracks = {}
        tracks.update(follow(allBlobs, frame, blob, +1))
        tracks.update(follow(allBlobs, frame, blob, -1))
        tracks[frame] = (blob, 0.0)

        for frames, (blob, histDist) in tracks.items():
            conf = round(1.0 - histDist, 2)

            if frames not in present or conf > present[frames]['confidence']:
                present[frames] = {
                    'x' : int(blob['centerX']), 
                    'y' : int(blob['centerY']), 
                    'box' : blob['box'], 
                    'source' : 'anchorsList' if frames == frame else 'blobTracking',
                    'confidence' : conf
                }

        sortedFrames = sorted(present)

        segments, start = [], None

        for k, frame in enumerate(sortedFrames):
            if start is None:
                start = frame

            last = (k == len(sortedFrames) - 1) or sortedFrames[k+1] != frame + 1

            if last:
                segments.append((start, frame))
                start = None

        for start, end in segments:
            hasAnchor = any(present[i]['source'] == 'anchorsList' for i in range(start, end + 1))

            if (end - start + 1) < MIN_SEGMENT and not hasAnchor:
                for frame in range(start, end + 1):
                    present.pop(frame)

        segments = [(start, end) for start, end in segments if any(frame in present for frame in range(start, end + 1))]

        with open ('frames.csv', 'w', newline = "") as file:
            w = csv.writer(file)
            w.writerow(['frame', 'timeS', 'staffPresent', 'x', 'y', 'boxX', 'boxY', 'boxWidth', 'boxHeight', 'source', 'confidence'])

            for frame in range(nFrames):
                p = present.get(frame)

                if p:
                    boxX, boxY, boxWidth, boxHeight = p['box']
                    w.writerow([frame, round(frame / fps, 2), 1, p['x'], p['y'], boxX, boxY, boxWidth, boxHeight, p['source'], p['confidence']])
                else:
                     w.writerow([frame, round(frame / fps, 2), 0, '', '', '', '', '', '', '', ''])

        with open('segments.csv', 'w', newline="") as file:
            w = csv.writer(file)
            w.writerow(['startFrame', 'endFrame', 'startS', 'endS', 'nFrames'])

            for start, end in segments:
                w.writerow([start, end, round(start / fps, 2), round(end / fps, 2), end - start + 1])

        cap = cv2.VideoCapture(VIDEO)
        out = cv2.VideoWriter('annotated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT))

        f = 0

        while True: 
            ready, frame = cap.read()

            if not ready:
                break

            p = present.get(f)

            if p: 
                boxX, boxY, boxWidth, boxHeight = p['box']
                cv2.rectangle(frame, (boxX, boxY), (boxX + boxWidth, boxY + boxHeight), (0, 255, 0), 3)
                cv2.putText(frame, 'STAFF', (boxX, max(boxY - 8, 15)), 0, 0.8, (0, 255, 0), 2)

            out.write(frame)
            f += 1

        out.release()

if __name__ == '__main__':
    main()


