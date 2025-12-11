"""
3D Object Tracking with Extended Kalman Filter

Simulates tracking a moving object in 3D space using noisy position sensors.
The object follows a helix path, and we add lots of noise to the measurements
to see how well the EKF can smooth things out.

We also test a 5-second sensor outage where the filter has to predict blindly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation setup
dt = 0.1
total_time = 30.0
num_steps = int(total_time / dt)

# Generate the true trajectory - a nice 3D helix
true_path = np.zeros((num_steps, 6))  # x, y, z, vx, vy, vz
true_path[0] = [0, 0, 0, 5.0, 2.0, 5.0]  # start with some velocity

for t in range(1, num_steps):
    x, y, z, vx, vy, vz = true_path[t-1]

    # Move based on velocity
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_z = z + vz * dt

    # Rotate the velocity a bit to create a helix
    new_vx = vx - 0.2 * vy * dt
    new_vy = vy + 0.2 * vx * dt
    new_vz = vz  # z velocity stays constant

    true_path[t] = [new_x, new_y, new_z, new_vx, new_vy, new_vz]

# Create noisy sensor measurements (only position, not velocity)
# Using high noise (1.5m) to really stress-test the filter
noise_level = 1.5
measurements = true_path[:, :3] + np.random.randn(num_steps, 3) * noise_level

# Set up the EKF
ekf_path = np.zeros((num_steps, 6))
# Start with a slightly wrong initial guess
ekf_path[0] = true_path[0] + np.random.randn(6) * [1, 1, 1, 0.5, 0.5, 0.5]

# Covariance matrix - our uncertainty about the state
P = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
uncertainty_history = np.zeros(num_steps)
uncertainty_history[0] = np.trace(P)

# Process noise - uncertainty in our motion model
Q = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])

# Measurement noise - uncertainty in our sensors
R = np.eye(3) * noise_level**2

# State transition matrix (how state evolves over time)
# Includes the rotation coupling that creates the helix
omega = 0.2
F = np.array([
    [1, 0, 0, dt, 0, 0],
    [0, 1, 0, 0, dt, 0],
    [0, 0, 1, 0, 0, dt],
    [0, 0, 0, 1, -omega*dt, 0],
    [0, 0, 0, omega*dt, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Measurement matrix - we only see position, not velocity
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

# Sensor outage from step 150 to 200 (5 seconds of blindness)
outage_start = 150
outage_end = 200

# Run the filter
for t in range(1, num_steps):
    # PREDICT: where do we think we are now?
    predicted = F @ ekf_path[t-1]
    P_predicted = F @ P @ F.T + Q

    # UPDATE: do we have a measurement to correct with?
    sensor_working = not (outage_start <= t < outage_end)

    if sensor_working:
        # We have a measurement - use it!
        z = measurements[t]
        error = z - H @ predicted
        S = H @ P_predicted @ H.T + R
        K = P_predicted @ H.T @ np.linalg.inv(S)

        ekf_path[t] = predicted + K @ error
        P = (np.eye(6) - K @ H) @ P_predicted
    else:
        # No measurement - just go with prediction
        ekf_path[t] = predicted
        P = P_predicted

    uncertainty_history[t] = np.trace(P)

# Calculate errors
raw_error = np.sqrt(np.sum((measurements - true_path[:, :3])**2, axis=1))
ekf_error = np.sqrt(np.sum((ekf_path[:, :3] - true_path[:, :3])**2, axis=1))

rmse_raw = np.sqrt(np.mean(raw_error**2))
rmse_ekf = np.sqrt(np.mean(ekf_error**2))
rmse_before = np.sqrt(np.mean(ekf_error[:outage_start]**2))
rmse_during = np.sqrt(np.mean(ekf_error[outage_start:outage_end]**2))
rmse_after = np.sqrt(np.mean(ekf_error[outage_end:]**2))

print("=" * 60)
print("3D TRACKING RESULTS")
print("=" * 60)
print(f"Raw measurements RMSE: {rmse_raw:.3f} m")
print(f"EKF estimate RMSE:     {rmse_ekf:.3f} m")
print(f"Improvement:           {(rmse_raw - rmse_ekf) / rmse_raw * 100:.1f}%")
print()
print("During sensor outage (5 seconds):")
print(f"  Before: {rmse_before:.3f} m")
print(f"  During: {rmse_during:.3f} m")
print(f"  After:  {rmse_after:.3f} m")
print("=" * 60)


# Plot results
fig = plt.figure(figsize=(14, 10))

# 3D trajectory
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(*true_path[:, :3].T, 'g-', lw=3, label='True path')

# Only show measurements when sensor was on
visible = np.ones(num_steps, dtype=bool)
visible[outage_start:outage_end] = False
ax.scatter(*measurements[visible].T, c='red', s=10, alpha=0.4, label='Measurements')

ax.plot(*ekf_path[:, :3].T, 'b-', lw=2, label='EKF estimate')
ax.plot(*ekf_path[outage_start:outage_end, :3].T, 'orange', lw=3, label='During outage')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory')
ax.legend()

# Error over time
ax = fig.add_subplot(2, 2, 2)
time = np.arange(num_steps) * dt
ax.plot(time, raw_error, 'r--', alpha=0.5, label='Raw measurements')
ax.plot(time, ekf_error, 'b-', lw=2, label='EKF')
ax.axvspan(outage_start*dt, outage_end*dt, alpha=0.3, color='orange', label='Sensor outage')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position error (m)')
ax.set_title('Error Over Time')
ax.legend()
ax.grid(True)

# Uncertainty
ax = fig.add_subplot(2, 2, 3)
ax.plot(time, uncertainty_history, 'purple', lw=2)
ax.axvspan(outage_start*dt, outage_end*dt, alpha=0.3, color='orange', label='Sensor outage')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Uncertainty')
ax.set_title('Filter Uncertainty')
ax.legend()
ax.grid(True)

# RMSE comparison
ax = fig.add_subplot(2, 2, 4)
labels = ['Raw\nmeasurements', 'EKF\noverall', 'Before\noutage', 'During\noutage', 'After\noutage']
values = [rmse_raw, rmse_ekf, rmse_before, rmse_during, rmse_after]
colors = ['red', 'blue', 'green', 'orange', 'cyan']
bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('RMSE (m)')
ax.set_title('Error Comparison')
ax.grid(True, axis='y')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('3d_simulation_output.png', dpi=200)
print("\nSaved: 3d_simulation_output.png")
plt.show()
