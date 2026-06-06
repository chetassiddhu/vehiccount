# 🚗 Real-Time Vehicle Detection and Traffic Analysis System

A production-ready computer vision system that detects, tracks, and counts vehicles in real-time using YOLOv8 and region-of-interest masking — built for intelligent traffic monitoring and congestion analysis.

---

## 🎯 Overview

This project implements a robust traffic analysis pipeline capable of accurately counting vehicles across multiple lanes using masked region-of-interest (ROI) detection. Designed for real-world deployment on traffic cameras, it provides actionable insights into traffic flow and congestion patterns.

**Achieved 73% counting accuracy** on real-world multi-lane traffic footage.

---

## ✨ Features

- 🔍 Real-time vehicle detection using **YOLOv8**
- 🎭 **Mask-based ROI filtering** to eliminate false detections outside the road zone
- 🛣️ **Multi-lane vehicle counting** with region-of-interest monitoring
- 📊 Live traffic flow and congestion pattern analysis
- 🎯 Object tracking for accurate per-vehicle counting (no double counts)
- 🧩 Modular pipeline — easy to plug into any camera feed
- 📹 Video file and live stream support

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Detection Model | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV (cv2) |
| Dataset | Roboflow |
| Masking | Custom NumPy mask |
| Tracking | SORT / ByteTrack |

---

## 📁 Project Structure

```
VehicleDetection/
│
├── videos/                  # Input traffic footage
├── masks/                   # ROI mask images per camera angle
├── runs/                    # YOLOv8 training/inference outputs
│
├── train.py                 # YOLOv8 model training script
├── detect.py                # Real-time detection + counting pipeline
├── tracker.py               # Vehicle tracking logic
├── mask_generator.py        # ROI mask creation utility
└── main.py                  # Main entry point
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/chetassiddhu/VehicleDetection.git
cd VehicleDetection
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python numpy roboflow
```

### 3. Download dataset from Roboflow
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("vehicle-detection")
dataset = project.version(1).download("yolov8")
```

### 4. Train the model (optional — pretrained weights included)
```bash
python train.py --data data.yaml --epochs 50 --imgsz 640
```

### 5. Run detection on video
```bash
python main.py --source videos/traffic.mp4 --mask masks/road_mask.png
```

---

## 🎭 How Masking Works

A custom binary mask is applied over each frame to restrict detection to the road area only — eliminating false positives from sidewalks, buildings, and sky.

```python
# Apply mask to frame
mask = cv2.imread("masks/road_mask.png", cv2.IMREAD_GRAYSCALE)
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
```

This significantly improves counting precision in complex urban environments.

---

## 🚀 How It Works

1. Video feed is loaded (file or live camera)
2. ROI mask is applied to each frame
3. YOLOv8 detects vehicles in the masked region
4. Tracker assigns unique IDs to each detected vehicle
5. Crossing line logic counts vehicles per lane
6. Live stats displayed as overlay on video feed

---

## 📊 Performance

| Metric | Result |
|---|---|
| Counting Accuracy | 73% on real-world footage |
| Model | YOLOv8n (nano) |
| Dataset | Custom — via Roboflow |
| Input | Video file / Live stream |

---

## 🔮 Future Improvements

- [ ] Speed estimation per vehicle
- [ ] Vehicle type classification (car, truck, bike)
- [ ] Dashboard with live traffic stats
- [ ] Integration with traffic signal control systems

---

## 👨‍💻 Author

**Chetas Siddhu**  
B.Tech + M.Tech (AI/ML) — Gautam Buddha University  
Research Intern @ IIT Roorkee | JNU | GGSIPU  
[LinkedIn](https://www.linkedin.com/in/chetas-siddhu-361578271) • [GitHub](https://github.com/chetassiddhu)
