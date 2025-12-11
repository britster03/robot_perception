"""
2D Robot Localization using Extended Kalman Filter

Simulates a robot driving in a circle while using two sensors:
1. IMU - gives velocity (but drifts over time)
2. Camera - spots a landmark and measures distance/angle to it

The EKF fuses both sensors to keep track of where the robot actually is.
We also test what happens when the camera stops working for 10 seconds.
"""

import numpy as np
import matplotlib.pyplot as plt

# How long to run the simulation
dt = 0.1  # seconds per step
total_time = 50.0
num_steps = int(total_time / dt)

# There's a landmark at this position that the camera can see
landmark = np.array([10.0, 10.0])

# Robot starts at origin, facing 45 degrees
true_path = np.zeros((num_steps, 3))  # x, y, heading
true_path[0] = [0, 0, np.pi/4]

# Robot moves forward at 1 m/s while turning slowly (makes a circle)
commands = np.zeros((num_steps, 2))  # velocity, turn rate
commands[:, 0] = 1.0  # forward speed
commands[:, 1] = 0.1  # turning speed

# Add realistic noise to IMU readings
imu_noise = np.array([0.2, 0.1])  # noise in velocity and turn rate
noisy_commands = commands + np.random.randn(num_steps, 2) * imu_noise

# Camera noise when measuring the landmark
camera_noise = np.array([0.5, 0.1])  # noise in distance and angle

# Simulate the robot's actual movement
for t in range(1, num_steps):
    x, y, heading = true_path[t-1]
    speed, turn = commands[t]

    # Simple motion: move forward, then turn
    true_path[t, 0] = x + speed * dt * np.cos(heading)
    true_path[t, 1] = y + speed * dt * np.sin(heading)
    true_path[t, 2] = heading + turn * dt


def predict_motion(state, cmd):
    """Where will the robot be after executing this command?"""
    x, y, heading = state
    speed, turn = cmd

    new_x = x + speed * dt * np.cos(heading)
    new_y = y + speed * dt * np.sin(heading)
    new_heading = heading + turn * dt

    return np.array([new_x, new_y, new_heading])


def predict_camera(state):
    """What would the camera see from this position?"""
    x, y, heading = state

    # Vector from robot to landmark
    dx = landmark[0] - x
    dy = landmark[1] - y

    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) - heading  # relative to robot's heading

    return np.array([distance, angle])


# Set up the EKF
ekf_path = np.zeros((num_steps, 3))
ekf_path[0] = true_path[0]  # start at true position

# How uncertain we are (starts small, grows during outages)
P = np.diag([0.1, 0.1, 0.1])
uncertainty_history = np.zeros(num_steps)
uncertainty_history[0] = np.trace(P)

# Process noise - how much we trust the motion model
Q = np.diag([0.05**2, 0.05**2, 0.03**2])

# Measurement noise - how much we trust the camera
R = np.diag([camera_noise[0]**2, camera_noise[1]**2])

# Camera breaks from step 250 to 350 (that's 10 seconds)
outage_start = 250
outage_end = 350

# Run the filter
for t in range(1, num_steps):
    # PREDICT: use IMU to guess where we moved
    predicted = predict_motion(ekf_path[t-1], noisy_commands[t])

    # Jacobian of motion model (how small changes in state affect the prediction)
    heading = ekf_path[t-1, 2]
    speed = noisy_commands[t, 0]
    F = np.array([
        [1, 0, -speed * dt * np.sin(heading)],
        [0, 1,  speed * dt * np.cos(heading)],
        [0, 0, 1]
    ])

    P_predicted = F @ P @ F.T + Q

    # UPDATE: correct our guess using camera (if it's working)
    camera_working = not (outage_start <= t < outage_end)

    if t % 5 == 0 and camera_working:  # camera updates every 5 steps
        # Simulate what the camera actually measured
        true_reading = predict_camera(true_path[t])
        noisy_reading = true_reading + np.random.randn(2) * camera_noise

        # Jacobian of measurement model
        dx = landmark[0] - predicted[0]
        dy = landmark[1] - predicted[1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)

        H = np.array([
            [-dx/dist, -dy/dist, 0],
            [dy/dist_sq, -dx/dist_sq, -1]
        ])

        # Kalman filter math
        expected = predict_camera(predicted)
        error = noisy_reading - expected
        S = H @ P_predicted @ H.T + R
        K = P_predicted @ H.T @ np.linalg.inv(S)

        final = predicted + K @ error
        P = (np.eye(3) - K @ H) @ P_predicted
    else:
        # No camera - just use our prediction
        final = predicted
        P = P_predicted

    ekf_path[t] = final
    uncertainty_history[t] = np.trace(P)


# Compare against dead reckoning (what if we only used IMU?)
dead_reckoning = np.zeros((num_steps, 3))
dead_reckoning[0] = true_path[0]
for t in range(1, num_steps):
    dead_reckoning[t] = predict_motion(dead_reckoning[t-1], noisy_commands[t])

# Calculate how well we did
dr_error = np.sqrt(np.sum((dead_reckoning[:, :2] - true_path[:, :2])**2, axis=1))
ekf_error = np.sqrt(np.sum((ekf_path[:, :2] - true_path[:, :2])**2, axis=1))

rmse_dr = np.sqrt(np.mean(dr_error**2))
rmse_ekf = np.sqrt(np.mean(ekf_error**2))
rmse_before = np.sqrt(np.mean(ekf_error[:outage_start]**2))
rmse_during = np.sqrt(np.mean(ekf_error[outage_start:outage_end]**2))
rmse_after = np.sqrt(np.mean(ekf_error[outage_end:]**2))

print("=" * 60)
print("2D LOCALIZATION RESULTS")
print("=" * 60)
print(f"Dead reckoning RMSE: {rmse_dr:.3f} m")
print(f"EKF RMSE:            {rmse_ekf:.3f} m")
print(f"Improvement:         {(rmse_dr - rmse_ekf) / rmse_dr * 100:.1f}%")
print()
print("During camera outage (10 seconds):")
print(f"  Before: {rmse_before:.3f} m")
print(f"  During: {rmse_during:.3f} m")
print(f"  After:  {rmse_after:.3f} m")
print("=" * 60)


# Plot everything
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trajectory comparison
ax = axes[0, 0]
ax.plot(true_path[:, 0], true_path[:, 1], 'g-', lw=3, label='True path')
ax.plot(dead_reckoning[:, 0], dead_reckoning[:, 1], 'r--', label='IMU only (drifts!)')
ax.plot(ekf_path[:, 0], ekf_path[:, 1], 'b-', lw=2, label='EKF estimate')
ax.plot(ekf_path[outage_start:outage_end, 0], ekf_path[outage_start:outage_end, 1],
        'orange', lw=3, label='During outage')
ax.plot(*landmark, 'k^', ms=12, label='Landmark')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Robot Trajectory')
ax.legend()
ax.grid(True)
ax.axis('equal')

# Error over time
ax = axes[0, 1]
time = np.arange(num_steps) * dt
ax.plot(time, dr_error, 'r--', alpha=0.6, label='Dead reckoning')
ax.plot(time, ekf_error, 'b-', lw=2, label='EKF')
ax.axvspan(outage_start*dt, outage_end*dt, alpha=0.3, color='orange', label='Camera outage')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position error (m)')
ax.set_title('Error Over Time')
ax.legend()
ax.grid(True)

# Uncertainty
ax = axes[1, 0]
ax.plot(time, uncertainty_history, 'purple', lw=2)
ax.axvspan(outage_start*dt, outage_end*dt, alpha=0.3, color='orange', label='Camera outage')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Uncertainty')
ax.set_title('Filter Uncertainty (grows during outage)')
ax.legend()
ax.grid(True)

# RMSE comparison
ax = axes[1, 1]
labels = ['Dead\nreckoning', 'EKF\noverall', 'Before\noutage', 'During\noutage', 'After\noutage']
values = [rmse_dr, rmse_ekf, rmse_before, rmse_during, rmse_after]
colors = ['red', 'blue', 'green', 'orange', 'cyan']
bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('RMSE (m)')
ax.set_title('Error Comparison')
ax.grid(True, axis='y')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('2d_simulation_output.png', dpi=200)
print("\nSaved: 2d_simulation_output.png")
plt.show()
