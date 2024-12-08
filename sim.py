import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the trained model
model = tf.keras.models.load_model("projectile_model.h5")

# Get user inputs
v0 = float(input("Enter the initial velocity (m/s): "))
theta = float(input("Enter the launch angle (degrees): "))

# Constants
g = 9.8  # Acceleration due to gravity (m/s^2)
dt = 0.05  # Time step (s)

# Convert angle to radians
theta_rad = np.radians(theta)

# Simulate the projectile motion
t = 0
trajectory = []

while True:
    # Prepare input for the model
    inputs = np.array([[v0, theta_rad, t]])  # Input format: [v0, theta (radians), t]
    # Predict position
    x, y = model.predict(inputs, verbose=0)[0]

    # Debugging: print predictions
   # print(f"Time: {t:.2f}, Predicted x: {x:.2f}, Predicted y: {y:.2f}")

    # Append the trajectory point
    trajectory.append((x, y))

    # Stop if the projectile hits the ground
    if y < 0:
        break

    # Increment time
    t += dt

    # Safeguard to prevent infinite loop
    if len(trajectory) > 1000:
        raise RuntimeError("Simulation exceeded maximum steps. Check the model.")

# Convert trajectory to a numpy array
trajectory = np.array(trajectory)

# Check for valid trajectory
if len(trajectory) == 0:
    raise ValueError("The trajectory is empty. Check the model predictions or inputs.")

# Ensure the plot fits the data
x_max = max(trajectory[:, 0]) * 1.1
y_max = max(trajectory[:, 1]) * 1.1

# Plot the trajectory with animation
fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size for better visibility
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_title("Projectile Motion Simulation")
ax.set_xlabel("Horizontal Distance (m)")
ax.set_ylabel("Vertical Distance (m)")

# Initialize the moving point and trajectory line
point, = ax.plot([], [], 'ro', markersize=5)  # The moving projectile
path, = ax.plot([], [], 'b-', lw=2)  # The trajectory line


# Update function for animation
def update(frame):
    point.set_data(trajectory[frame, 0], trajectory[frame, 1])
    path.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
    return point, path


# Create animation
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=20, blit=True)

# Display the plot
plt.tight_layout()  # Ensure everything fits nicely
plt.show()
