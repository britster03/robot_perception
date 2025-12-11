#!/usr/bin/env python3
"""
Real-Time ARUco Marker Tracking with EKF

Uses your webcam to track a printed ARUco marker in 3D space.
The raw pose estimates are noisy, so we use an Extended Kalman Filter
to smooth them out and keep tracking even when the marker disappears.

Controls:
  q - quit
  r - reset the filter
"""

import sys
import time
import os
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# What marker are we looking for?
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()
MARKER_ID = 0
MARKER_SIZE = 0.2032  # 8 inches in meters

# Camera settings (approximate for a typical 1080p webcam)
CAMERA_MATRIX = np.array([
    [1400, 0, 960],
    [0, 1400, 540],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# EKF tuning
Q_ACCEL = 0.05   # how much random acceleration we expect
R_MEAS = 0.002   # how noisy the measurements are


class EKF:
    """
    Simple Extended Kalman Filter for 3D tracking.

    Tracks position and velocity: [x, y, z, vx, vy, vz]
    Assumes constant velocity motion model.
    """

    def __init__(self, dt):
        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        self.P = np.eye(6)

        # We'll update F and Q when dt changes
        self._update_matrices(dt)

        # We measure position only
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.eye(3) * R_MEAS

    def _update_matrices(self, dt):
        # State transition: position = position + velocity * dt
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        # Process noise from random acceleration
        G = np.zeros((6, 3))
        G[:3, :3] = 0.5 * dt**2 * np.eye(3)
        G[3:, :3] = dt * np.eye(3)
        self.Q = Q_ACCEL * (G @ G.T)

    def set_dt(self, dt):
        self.dt = dt
        self._update_matrices(dt)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


def open_camera():
    """Try to open the webcam. Handles macOS quirks."""
    if sys.platform == "darwin":
        # Mac needs AVFoundation
        for idx in [0, 1]:
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
            cap.release()

    # Try generic approach
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()

    return None


def main():
    # Open the camera
    cap = open_camera()
    if cap is None:
        print("Can't open camera!")
        print("- Close other apps using the camera (Zoom, FaceTime, etc)")
        print("- Check System Settings > Privacy > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camera opened successfully")

    # Set up detection and filtering
    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    ekf = EKF(1/30)
    initialized = False

    # Keep track of measurements for plotting later
    measurements = []
    estimates = []
    estimates_at_meas = []

    last_time = time.perf_counter()

    print("Press 'q' to quit, 'r' to reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate actual time step
        now = time.perf_counter()
        dt = min(max(now - last_time, 0.001), 0.2)  # keep it reasonable
        last_time = now
        ekf.set_dt(dt)

        # Look for markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False

        if ids is not None and MARKER_ID in ids.flatten():
            # Found our marker! Get its pose
            idx = np.where(ids.flatten() == MARKER_ID)[0][0]
            marker_corners = corners[idx]

            # Define marker corners in 3D (marker coordinate frame)
            half = MARKER_SIZE / 2
            obj_points = np.array([
                [-half, half, 0],
                [half, half, 0],
                [half, -half, 0],
                [-half, -half, 0]
            ], dtype=np.float32)

            # Solve for pose
            ok, rvec, tvec = cv2.solvePnP(
                obj_points,
                marker_corners.astype(np.float32),
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok:
                found = True
                pos = tvec.flatten()
                measurements.append(pos.copy())

                # Initialize filter from first measurement
                if not initialized:
                    ekf.x[:3, 0] = pos
                    ekf.x[3:, 0] = 0
                    ekf.P = np.eye(6) * 0.1
                    initialized = True
                    print(f"Started tracking at {pos}")

                # Run the filter
                ekf.predict()
                ekf.update(tvec)
                estimates_at_meas.append(ekf.x[:3, 0].copy())

                # Draw detection
                aruco.drawDetectedMarkers(frame, [marker_corners.reshape(1, 4, 2)])
                try:
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)
                except:
                    pass

        # No marker? Just predict (with velocity decay to prevent drift)
        if not found and initialized:
            ekf.predict()
            ekf.x[3:, 0] *= 0.95  # slow down gradually

        if initialized:
            estimates.append(ekf.x[:3, 0].copy())

        # Draw info overlay
        state = ekf.x.flatten()
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(frame, f"Detected: {found}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if initialized:
            cv2.putText(frame, f"Position: x={state[0]:.2f} y={state[1]:.2f} z={state[2]:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Velocity: vx={state[3]:.2f} vy={state[4]:.2f} vz={state[5]:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("ARUco Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ekf = EKF(1/30)
            initialized = False
            measurements.clear()
            estimates.clear()
            estimates_at_meas.clear()
            print("Reset!")

    cap.release()
    cv2.destroyAllWindows()

    # Show results if we collected data
    if len(estimates) < 10:
        print("Not enough data to plot")
        return

    est = np.array(estimates)
    meas = np.array(measurements) if measurements else None
    est_at_meas = np.array(estimates_at_meas) if estimates_at_meas else None

    print(f"\nCollected {len(estimates)} estimates, {len(measurements)} measurements")

    # Calculate metrics
    if meas is not None and len(meas) > 10:
        # How much did we smooth the data?
        meas_noise = np.std(np.diff(meas, n=2, axis=0))
        est_noise = np.std(np.diff(est_at_meas, n=2, axis=0))
        noise_reduction = (1 - est_noise / meas_noise) * 100

        # How jumpy is the trajectory?
        meas_jitter = np.mean(np.linalg.norm(np.diff(meas, axis=0), axis=1))
        est_jitter = np.mean(np.linalg.norm(np.diff(est_at_meas, axis=0), axis=1))
        jitter_reduction = (1 - est_jitter / meas_jitter) * 100

        # Detection rate
        detection_rate = len(meas) / len(est) * 100

        print(f"\nResults:")
        print(f"  Noise reduction: {noise_reduction:.0f}%")
        print(f"  Jitter reduction: {jitter_reduction:.0f}%")
        print(f"  Detection rate: {detection_rate:.0f}%")
        print(f"  Prediction-only frames: {100 - detection_rate:.0f}%")

    # Plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("ARUco + EKF Tracking Results", fontsize=14, fontweight='bold')

    # 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(*est.T, 'b-', lw=1.5, label='EKF estimate')
    if meas is not None:
        ax.scatter(*meas[::max(1, len(meas)//100)].T, c='red', s=15, alpha=0.5, label='Measurements')
        ax.scatter(*meas[0], c='green', s=100, marker='^', label='Start')
        ax.scatter(*meas[-1], c='purple', s=100, marker='s', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()

    # X over time
    ax = fig.add_subplot(2, 2, 2)
    frames = np.arange(len(est))
    ax.plot(frames, est[:, 0], 'b-', lw=1.5, label='EKF')
    if meas is not None:
        meas_frames = np.linspace(0, len(est)-1, len(meas))
        ax.scatter(meas_frames, meas[:, 0], c='red', s=8, alpha=0.3, label='Raw')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X (m)')
    ax.set_title('X Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y over time
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(frames, est[:, 1], 'b-', lw=1.5, label='EKF')
    if meas is not None:
        ax.scatter(meas_frames, meas[:, 1], c='red', s=8, alpha=0.3, label='Raw')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y (m)')
    ax.set_title('Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z over time
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(frames, est[:, 2], 'b-', lw=1.5, label='EKF')
    if meas is not None:
        ax.scatter(meas_frames, meas[:, 2], c='red', s=8, alpha=0.3, label='Raw')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Z (m)')
    ax.set_title('Z Position (Depth)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aruco_tracking_output.png', dpi=150)
    print("\nSaved: aruco_tracking_output.png")
    plt.show()


if __name__ == "__main__":
    main()
