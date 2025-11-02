import cv2
import math
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import *
print("guru ji")

# Video load
cap = cv2.VideoCapture("/home/chetas-siddhu/Downloads/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4")

# Model load
model = YOLO("../YOLO-weights/yolov8n.pt")

# COCO classes
classnames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Mask grayscale me load
mask = cv2.imread("Mask.png", 0)
#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# Line (x1, y1, x2, y2)
#limits = [600, 526, 1475, 526]   # ek horizontal line
limits = [100, 1350, 2600, 1350]   # Horizontal line, thoda right aur niche

totalcount = []
while True:
    success, img = cap.read()
    if not success:
        break

    # Mask resize
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Masked region nikaalo
    imgregion = cv2.bitwise_and(img, img, mask=mask_resized)

    # Detection on masked region only
    results = model(imgregion, stream=True)
    detection = np.empty((0,5))
    for r in results:
        for box in r.boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classnames[cls]

            # sirf vehicle classes detect karo
            if currentclass in ["car", "motorcycle", "bus", "truck"] and conf > 0.3:
                #cvzone.putTextRect(imgregion, f'{currentclass} {conf}', (max(0, x1), max(30, y1)))
                cv2.rectangle(imgregion, (x1, y1), (x2, y2), (225, 0, 225), 3)
                #currentarry = np.empty([x1,y1,x2,y2,conf])
                currentarray = np.array([x1, y1, x2, y2, conf])

                detection = np.vstack((detection,currentarray))
    resulttracker = tracker.update(detection)
    cv2.line(imgregion,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,225),1)
    for result in resulttracker:
        x1,y1,x2,y2,ID = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cv2.rectangle(imgregion, (x1, y1), (x2, y2), (225, 0, 0), 5)
        cvzone.putTextRect(imgregion, f'{ID}', (max(0, x1), max(30, y1)))
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(imgregion, (int(cx), int(cy)), 10, (225, 0, 225), -1)
        if limits[0]<cx<limits[2] and limits[1]-15 < cy <limits[1] + 15:
            if totalcount.count(ID)==0:
                totalcount.append(ID)
                cv2.line(imgregion, (limits[0], limits[1]), (limits[2], limits[3]), (0, 225, 225), 25)

        cvzone.putTextRect(imgregion, f' count:{len(totalcount)}', (50,50))

    # Show output (only masked region detections)
    #cv2.imshow("Detections in Mask Region", imgregion)


    imgregion = cv2.resize(imgregion, (740, 540))  # apna custom size
    cv2.imshow("Detections in Mask Region", imgregion)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
