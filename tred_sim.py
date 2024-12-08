import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.8  # Acceleration due to gravity (m/s^2)
dt = 0.05  # Time step (s)

# User input for initial conditions
v0 = float(input("Enter the initial velocity (m/s): "))
theta = float(input("Enter the launch angle (degrees): "))
theta = np.radians(theta)  # Convert to radians

# Calculating initial velocities
vx = v0 * np.cos(theta)  # Horizontal velocity (m/s)
vy = v0 * np.sin(theta)  # Vertical velocity (m/s)
x, y = 0, 0  # Initial position (m)

# Lists to store results
x_vals, y_vals = [x], [y]
t = 0

# Simulation loop
while y >= 0:
    x += vx * dt
    y += vy * dt - 0.5 * g * dt**2
    vy -= g * dt
    x_vals.append(x)
    y_vals.append(y)
    t += dt

# Plotting setup
fig, ax = plt.subplots()
ax.set_xlim(0, max(x_vals) * 1.1)
ax.set_ylim(0, max(y_vals) * 1.1)
ax.set_title('Projectile Motion with Animation')
ax.set_xlabel('Horizontal Distance (m)')
ax.set_ylabel('Vertical Distance (m)')

point, = ax.plot([], [], 'ro')  # A single point for the animation
path, = ax.plot([], [], 'b-')   # The trajectory line

# Animation function
def update(frame):
    point.set_data(x_vals[frame], y_vals[frame])
    path.set_data(x_vals[:frame+1], y_vals[:frame+1])
    return point, path

# Create animation
ani = FuncAnimation(fig, update, frames=len(x_vals), interval=50, blit=True)

plt.show()
