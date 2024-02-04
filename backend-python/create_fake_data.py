import numpy as np
import random
import pickle

def generate_sine_curve_x_values(num_points=100, peak_value=10000):
    # Create a sine curve ranging from -pi/2 to pi/2
    x_values_sine = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Scale and shift the sine curve to fit the desired range (0 to 10000)
    x_values_scaled = (np.sin(x_values_sine) + 1) * (peak_value / 2)

    # Convert to floats with max 2 decimal places
    x_values_float = np.round(x_values_scaled, 2)

    return x_values_float

def generate_cos_curve_x_values(num_points=100, peak_value=10000):
    # Create a sine curve ranging from -pi/2 to pi/2
    x_values_sine = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Scale and shift the sine curve to fit the desired range (0 to 10000)
    x_values_scaled = (np.cos(x_values_sine) + 1) * (peak_value / 2)

    # Convert to floats with max 2 decimal places
    x_values_float = np.round(x_values_scaled, 2)

    return x_values_float

# Example usage:
data1 = generate_sine_curve_x_values(num_points=10000, peak_value=10000) + random.uniform(-10000, 10000)
data2 = generate_cos_curve_x_values(num_points=10000, peak_value=10000) + random.uniform(-10000, 10000)
data3 = np.linspace(10000, 0, 10000) + random.uniform(-10000, 10000)
data4 = np.linspace(0, 10000, 10000) + random.uniform(-10000, 10000)
data5 = generate_sine_curve_x_values(num_points=10000, peak_value=10000) + random.uniform(-10000, 10000)
data6 = generate_cos_curve_x_values(num_points=10000, peak_value=10000) + random.uniform(-10000, 10000)
data7 = np.linspace(10000, 0, 10000) + random.uniform(-10000, 10000)
data8 = np.linspace(0, 10000, 10000) + random.uniform(-10000, 10000)
for i in range(10000):
    if data1[i] < 0:
        data1[i] = 0
    if data2[i] < 0:
        data2[i] = 0
    if data3[i] < 0:
        data3[i] = 0
    if data4[i] < 0:
        data4[i] = 0
    if data5[i] < 0:
        data5[i] = 0
    if data6[i] < 0:
        data6[i] = 0
    if data7[i] < 0:
        data7[i] = 0
    if data8[i] < 0:
        data8[i] = 0

l = []
for i in range(10000):
    l.append([data1[i], data2[i], data3[i], data4[i], data5[i], data6[i], data7[i], data8[i]])

with open("fake_data.pickle", 'wb') as file:
    pickle.dump(l, file)
