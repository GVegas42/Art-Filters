import numpy as np

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