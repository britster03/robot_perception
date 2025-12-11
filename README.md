# 3D Object Tracking with Extended Kalman Filter

A robot perception project that uses Extended Kalman Filtering (EKF) to track objects in 3D space. The filter smooths out noisy sensor measurements and keeps tracking even when sensors temporarily fail.

## What This Project Does

1. **2D Simulation** - A robot drives in a circle using IMU + camera sensors. Shows how EKF beats dead reckoning.
2. **3D Simulation** - Track an object moving in a helix pattern with noisy position measurements.
3. **Real Webcam Tracking** - Track a printed ARUco marker in real-time using your laptop camera.

All three demonstrate how EKF handles:
- Noisy measurements (the filter smooths them out)
- Sensor outages (the filter keeps predicting)

## Requirements

```
Python 3.8+
numpy
opencv-python (with aruco module)
matplotlib
```

## Quick Install

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy opencv-python opencv-contrib-python matplotlib
```

## How to Run

### 1. 2D Simulation (no hardware needed)

```bash
python 2d_simulation.py
```

This simulates a robot driving in a circle. You'll see:
- Green line = true path
- Red dashed = IMU-only (drifts badly!)
- Blue = EKF estimate (much better)
- Orange = during camera outage

Output: `2d_simulation_output.png`

### 2. 3D Simulation (no hardware needed)

```bash
python 3d_simulation.py
```

Tracks an object moving in a 3D helix with very noisy measurements. The EKF recovers the smooth trajectory from the noise.

Output: `3d_simulation_output.png`

### 3. Real-Time Webcam Tracking

First, print an ARUco marker:
1. Go to: https://chev.me/arucogen/
2. Select Dictionary: 4x4 (50 markers)
3. Marker ID: 0
4. Size: 200mm (or 8 inches)
5. Print it out

Then run:

```bash
python aruco_detect_v1.py
```

Controls:
- `q` = quit
- `r` = reset filter

Move the marker around in front of your camera. The blue trail is the filtered position, red/orange dots are raw measurements.

Output: `aruco_tracking_output.png` (saved when you quit)

## Project Structure

```
robot_perception_project/
├── 2d_simulation.py      # 2D robot localization demo
├── 3d_simulation.py      # 3D object tracking demo
├── aruco_detect_v1.py    # Real-time webcam tracking
├── README.md             # This file
└── final_report_template/  # LaTeX report template
```

## Results Summary

| Phase | What We Tested | Improvement |
|-------|---------------|-------------|
| 2D Sim | IMU + Camera fusion | 91% RMSE reduction vs dead reckoning |
| 3D Sim | Noisy position filtering | 59% RMSE reduction |
| Webcam | Real-time ARUco tracking | 48% noise reduction, 31% jitter reduction |

The EKF also handled ~38% missing measurements in real-world testing by predicting through outages.

## Troubleshooting

**Camera won't open (Mac)**
- Close Zoom, FaceTime, or any other app using the camera
- Go to System Settings > Privacy & Security > Camera and allow Terminal/Python

**No markers detected**
- Make sure the marker is well-lit and not too far away
- The marker should be flat (not crumpled)
- Try marker ID 0 from the 4x4_50 dictionary

**Import errors**
- Make sure you installed `opencv-contrib-python` (not just `opencv-python`)

## Author

Ronit C. Virwani
