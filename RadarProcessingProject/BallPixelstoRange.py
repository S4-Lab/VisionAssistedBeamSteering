import numpy as np
from scipy.optimize import curve_fit

# Example: measured distances (meters)
distances = np.array([0.10, 0.15, 0.20, 0.25])  # replace with your actual distances in meters

# Measured pixel diameters from your image program
pixels = np.array([146, 99, 74, 59])  # replace with actual pixel measurements

# The model: Z = k / d_pixels
def model(d_pixels, k):
    return k / d_pixels

# Fit k
popt, _ = curve_fit(model, pixels, distances)
k = popt[0]
print(f"Fitted constant k = {k:.4f}")

# Function to predict distance from pixel diameter
def pixels_to_distance(d_pixels):
    return k / d_pixels

# Function to predict pixel diameter from distance
def distance_to_pixels(Z):
    return k / Z