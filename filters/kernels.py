import numpy as np

def get_sharpen():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def get_emboss():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

def get_outline():
    return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def get_gaussian_blur(size=5, sigma=1.0):
    # Use the Gaussian logic from your notebook
    pass