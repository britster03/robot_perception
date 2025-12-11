# --- add near your imports ---
import sys, cv2

def open_camera(preferred=(1, 0)):  # try Continuity/iPhone first, then built-in
    apis = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]
    for api in apis:
        for idx in preferred:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                print(f"[camera] opened idx={idx} api={api}")
                return cap
            cap.release()
    return None

