import numpy as np

def get_sharpen():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def get_emboss():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

def get_outline():
    return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def get_sobel_horizontal():
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ])

def get_sobel_vertical():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])

def get_laplacian_4():
    return np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

def get_laplacian_8():
    return np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

def get_laplacian():
    # Backward-compatible alias for the original Laplacian kernel.
    return get_laplacian_4()

def get_gaussian_blur(size=5, sigma=1.0):
    if size % 2 == 0:
        raise ValueError("Gaussian blur size must be an odd number.")

    radius = size // 2
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def get_log(size=7, sigma=1.0):
    if size % 2 == 0:
        raise ValueError("LoG size must be an odd number.")

    radius = size // 2
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)

    radius_squared = xx ** 2 + yy ** 2
    sigma_squared = sigma ** 2

    # Continuous LoG expression sampled on a square grid.
    log_kernel = ((radius_squared - (2 * sigma_squared)) / (sigma_squared ** 2)) * np.exp(
        -radius_squared / (2 * sigma_squared)
    )

    # Enforce near-zero response on flat regions by removing numerical bias.
    log_kernel -= np.mean(log_kernel)

    return log_kernel

def get_gabor(size=9, sigma=2.5, theta=np.pi / 4, wavelength=4.0, gamma=0.5, phase_offset=0.0):
    if size % 2 == 0:
        raise ValueError("Gabor filter size must be an odd number.")

    radius = size // 2
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)

    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

    gaussian_envelope = np.exp(
        -(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2 * sigma ** 2)
    )
    sinusoid = np.cos((2 * np.pi * x_theta / wavelength) + phase_offset)
    kernel = gaussian_envelope * sinusoid

    # Center the response and keep the convolution scale stable.
    kernel -= np.mean(kernel)
    normalization = np.sum(np.abs(kernel))
    if normalization != 0:
        kernel /= normalization

    return kernel


def get_median_oil_base(size=5):
    return {
        'type': 'median',
        'size': size,
    }

def get_all_filters():
    return {
        'sharpen': get_sharpen(),
        'emboss': get_emboss(),
        'outline': get_outline(),
        'sobel_horizontal': get_sobel_horizontal(),
        'sobel_vertical': get_sobel_vertical(),
        'laplacian': get_laplacian(),
        'laplacian_4': get_laplacian_4(),
        'laplacian_8': get_laplacian_8(),
        'gaussian_blur': get_gaussian_blur(),
        'log': get_log(),
        'gabor': get_gabor(),
        'median_oil_base': get_median_oil_base(),
    }