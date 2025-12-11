#!/bin/bash
# Run ARUco tracking without warning spam
python3 aruco_detect_v1.py 2>&1 | grep -v "drawFrameAxes"
