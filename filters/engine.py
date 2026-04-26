import numpy as np


def apply_art_filter(image, filter_spec):
    """Apply either a convolution kernel filter or a non-linear filter config."""
    if isinstance(filter_spec, np.ndarray):
        return apply_filter(image, filter_spec)

    if isinstance(filter_spec, dict):
        filter_type = filter_spec.get("type")
        if filter_type == "median":
            return apply_median_filter(image, size=int(filter_spec.get("size", 5)))

    raise TypeError(f"Unsupported filter specification: {type(filter_spec).__name__}")

def apply_filter(image, kernel):
    """
    Applies a filter to an image. Works for both Grayscale and RGB.
    """
    if len(image.shape) == 3:  # RGB Image
        # Apply the filter to each channel individually
        channels = []
        for i in range(3):
            channels.append(conv_fast(image[:, :, i], kernel))
        return np.stack(channels, axis=2)
    else:
        return conv_fast(image, kernel)


def apply_median_filter(image, size=5, padding_mode='reflect'):
    """Apply a median filter, commonly used as an oil-painting base smoothing step."""
    if size <= 0 or size % 2 == 0:
        raise ValueError("Median filter size must be a positive odd number.")

    if len(image.shape) == 3:
        channels = []
        for i in range(3):
            channels.append(median_filter_single_channel(image[:, :, i], size, padding_mode))
        return np.stack(channels, axis=2)

    return median_filter_single_channel(image, size, padding_mode)


def median_filter_single_channel(image, size, padding_mode):
    """Median filter implementation for one 2D channel."""
    Hi, Wi = image.shape
    pad = size // 2
    out = np.zeros((Hi, Wi))

    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode=padding_mode)

    for i in range(Hi):
        for j in range(Wi):
            patch = image_padded[i:i+size, j:j+size]
            out[i, j] = np.median(patch)

    return out
    
def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    ### END YOUR CODE
    return out

def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_h = Hk // 2
    pad_w = Wk // 2

    image_padded = zero_pad(image, pad_h, pad_w)
    kernel_flipped = np.flip(kernel).copy()

    for i in range(Hi):
        for j in range(Wi):
            patch = image_padded[i:i+Hk, j:j+Wk]
            out[i, j] = np.sum(patch * kernel_flipped)
    ### END YOUR CODE

    return out