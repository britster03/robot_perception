
# save as check_cv.py and run: python3 check_cv.py
import cv2, re
info = cv2.getBuildInformation()
print("AVFoundation present? ", bool(re.search(r'AVFoundation:\s+YES', info)))
print("\n=== Video I/O block (first 1200 chars) ===\n")
start = info.find("Video I/O:")
print(info[start:start+1200])

