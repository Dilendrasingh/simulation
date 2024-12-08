import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Constants
g = 9.8  # Acceleration due to gravity (m/s^2)
dt = 0.01  # Time step (s)

# Function to simulate projectile motion
def simulate_projectile(v0, theta):
    theta = np.radians(theta)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0, 0
    t = 0
    data = []
    while y >= 0:
        x += vx * dt
        y += vy * dt - 0.5 * g * dt**2
        vy -= g * dt
        t += dt
        data.append([v0, theta, t, x, y])  # Features: v0, theta, t | Labels: x, y
    return data

# Generate training data
projectile_data = []
for v0 in np.linspace(10, 100, 50):  # Initial velocities
    for theta in np.linspace(10, 80, 50):  # Launch angles
        projectile_data.extend(simulate_projectile(v0, theta))

# Convert data to numpy array
projectile_data = np.array(projectile_data)
X = projectile_data[:, :3]  # Features: v0, theta, t
y = projectile_data[:, 3:]  # Labels: x, y

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Outputs: x, y
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
model.save("projectile_model.h5")
print("Model saved as 'projectile_model.h5'")
