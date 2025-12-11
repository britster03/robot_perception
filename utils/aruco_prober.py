# aruco_probe.py  -> run:  python aruco_probe.py --cam 0
import sys, argparse, cv2
from cv2 import aruco

def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)  # fallback
    return cap

def make_params():
    p = aruco.DetectorParameters()
    # More forgiving for low light / screen glare
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 31
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant   = 7
    p.minMarkerPerimeterRate   = 0.02
    p.maxMarkerPerimeterRate   = 4.0
    p.minCornerDistanceRate    = 0.05
    p.minDistanceToBorder      = 1
    p.polygonalApproxAccuracyRate = 0.03
    p.cornerRefinementMethod   = aruco.CORNER_REFINE_SUBPIX
    p.markerBorderBits         = 1
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    cap = open_cam(args.cam)
    if not cap or not cap.isOpened():
        sys.exit("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = make_params()
    detector = aruco.ArucoDetector(dictionary, params)

    print("Press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # New API
        corners, ids, rejected = detector.detectMarkers(gray)

        # Legacy fallback if needed
        if ids is None:
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"IDs: {ids.flatten().tolist()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, "No markers", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("aruco probe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

