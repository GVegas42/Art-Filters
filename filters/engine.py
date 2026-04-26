import numpy as np


def apply_art_filter(image, filter_spec):
    """Apply either a convolution kernel filter or a non-linear filter config."""
    if isinstance(filter_spec, np.ndarray):
        return apply_filter(image, filter_spec)

    if isinstance(filter_spec, dict):
        filter_type = filter_spec.get("type")
        if filter_type == "median":
            return apply_median_filter(image, size=int(filter_spec.get("size", 5)))
        if filter_type == "kuwahara":
            return apply_kuwahara_filter(image, size=int(filter_spec.get("size", 5)))
        if filter_type == "gradient_magnitude":
            return apply_gradient_magnitude_filter(
                image,
                kernel_x=np.asarray(filter_spec["kernel_x"]),
                kernel_y=np.asarray(filter_spec["kernel_y"]),
                sketch_boost=bool(filter_spec.get("sketch_boost", False)),
                invert=bool(filter_spec.get("invert", False)),
            )
        if filter_type == "quantize_dither":
            return apply_quantization_dither_filter(
                image,
                levels=int(filter_spec.get("levels", 4)),
                dithering=str(filter_spec.get("dithering", "floyd_steinberg")),
                bayer_size=int(filter_spec.get("bayer_size", 4)),
                serpentine=bool(filter_spec.get("serpentine", True)),
            )

    raise TypeError(f"Unsupported filter specification: {type(filter_spec).__name__}")


def apply_quantization_dither_filter(image, levels=4, dithering="floyd_steinberg", bayer_size=4, serpentine=True):
    """Apply per-channel color quantization with optional dithering for retro looks."""
    if levels < 2:
        raise ValueError("levels must be >= 2.")

    if image.ndim == 2:
        image_work = image[..., np.newaxis]
        squeeze_output = True
    elif image.ndim == 3:
        image_work = image
        squeeze_output = False
    else:
        raise ValueError("Unsupported image shape for quantization/dithering.")

    image_work = np.clip(image_work.astype(np.float64), 0.0, 1.0)
    dithering = dithering.lower()

    if dithering == "none":
        out = quantize_per_channel(image_work, levels)
    elif dithering == "ordered_bayer":
        out = apply_ordered_bayer_dither(image_work, levels=levels, bayer_size=bayer_size)
    elif dithering == "floyd_steinberg":
        out = apply_floyd_steinberg_dither(image_work, levels=levels, serpentine=serpentine)
    else:
        raise ValueError("dithering must be one of: none, ordered_bayer, floyd_steinberg")

    if squeeze_output:
        return out[..., 0]
    return out


def quantize_per_channel(image, levels):
    """Uniformly quantize each channel independently to the provided number of levels."""
    scale = float(levels - 1)
    return np.clip(np.round(image * scale) / scale, 0.0, 1.0)


def apply_floyd_steinberg_dither(image, levels=4, serpentine=True):
    """Apply Floyd-Steinberg error diffusion on each channel."""
    h, w, c = image.shape
    work = image.copy()
    out = np.zeros_like(work)

    for y in range(h):
        left_to_right = (y % 2 == 0) or (not serpentine)
        if left_to_right:
            x_iter = range(w)
            neighbors = (
                (1, 0, 7.0 / 16.0),
                (-1, 1, 3.0 / 16.0),
                (0, 1, 5.0 / 16.0),
                (1, 1, 1.0 / 16.0),
            )
        else:
            x_iter = range(w - 1, -1, -1)
            neighbors = (
                (-1, 0, 7.0 / 16.0),
                (1, 1, 3.0 / 16.0),
                (0, 1, 5.0 / 16.0),
                (-1, 1, 1.0 / 16.0),
            )

        for x in x_iter:
            old_pixel = np.clip(work[y, x, :], 0.0, 1.0)
            new_pixel = quantize_per_channel(old_pixel, levels)
            out[y, x, :] = new_pixel
            error = old_pixel - new_pixel

            for dx, dy, weight in neighbors:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    work[ny, nx, :] += error * weight

    return np.clip(out, 0.0, 1.0)


def apply_ordered_bayer_dither(image, levels=4, bayer_size=4):
    """Apply ordered Bayer dithering before channel quantization."""
    matrix = get_bayer_matrix(bayer_size)
    h, w, _ = image.shape

    tiled = np.tile(
        matrix,
        (
            int(np.ceil(h / matrix.shape[0])),
            int(np.ceil(w / matrix.shape[1])),
        ),
    )[:h, :w]

    threshold = ((tiled + 0.5) / (matrix.size)) - 0.5
    amplitude = 1.0 / float(levels - 1)
    dithered = np.clip(image + (threshold[..., np.newaxis] * amplitude), 0.0, 1.0)
    return quantize_per_channel(dithered, levels)


def get_bayer_matrix(size):
    """Return a Bayer matrix of size 2, 4, or 8."""
    if size not in (2, 4, 8):
        raise ValueError("bayer_size must be one of 2, 4, or 8.")

    matrix = np.array([[0, 2], [3, 1]], dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block(
            [
                [4 * matrix + 0, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
    return matrix

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


def apply_gradient_magnitude_filter(image, kernel_x, kernel_y, sketch_boost=False, invert=False):
    """Apply combined edge magnitude from two orthogonal gradient kernels."""
    if len(image.shape) == 3:
        channels = []
        for i in range(3):
            grad_x = conv_fast(image[:, :, i], kernel_x)
            grad_y = conv_fast(image[:, :, i], kernel_y)
            channels.append(np.sqrt((grad_x ** 2) + (grad_y ** 2)))
        out = np.stack(channels, axis=2)
        if sketch_boost:
            return apply_sketch_boost(out, invert=invert)
        return out

    grad_x = conv_fast(image, kernel_x)
    grad_y = conv_fast(image, kernel_y)
    out = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    if sketch_boost:
        return apply_sketch_boost(out, invert=invert)
    return out


def apply_sketch_boost(image, invert=False, low_percentile=2.0, high_percentile=98.0):
    """Boost sketch readability by contrast stretching edge intensity."""
    if high_percentile <= low_percentile:
        raise ValueError("high_percentile must be greater than low_percentile.")

    if len(image.shape) == 3:
        low = np.percentile(image, low_percentile, axis=(0, 1), keepdims=True)
        high = np.percentile(image, high_percentile, axis=(0, 1), keepdims=True)
    else:
        low = np.percentile(image, low_percentile)
        high = np.percentile(image, high_percentile)

    scale = np.maximum(high - low, 1e-8)
    boosted = np.clip((image - low) / scale, 0.0, 1.0)

    if invert:
        boosted = 1.0 - boosted

    return boosted


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


def apply_kuwahara_filter(image, size=5, padding_mode='reflect'):
    """Apply a classic Kuwahara filter for painterly abstraction."""
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kuwahara filter size must be a positive odd number.")

    if len(image.shape) == 3:
        return kuwahara_filter_rgb(image, size=size, padding_mode=padding_mode)

    return kuwahara_filter_single_channel(image, size=size, padding_mode=padding_mode)


def kuwahara_filter_single_channel(image, size=5, padding_mode='reflect'):
    """Kuwahara filter for one 2D channel."""
    Hi, Wi = image.shape
    radius = size // 2
    out = np.zeros((Hi, Wi))

    image_padded = np.pad(image, ((radius, radius), (radius, radius)), mode=padding_mode)

    for i in range(Hi):
        for j in range(Wi):
            patch = image_padded[i:i+size, j:j+size]
            regions = [
                patch[0:radius + 1, 0:radius + 1],
                patch[0:radius + 1, radius:size],
                patch[radius:size, 0:radius + 1],
                patch[radius:size, radius:size],
            ]

            variances = [np.var(region) for region in regions]
            best_idx = int(np.argmin(variances))
            out[i, j] = np.mean(regions[best_idx])

    return out


def kuwahara_filter_rgb(image, size=5, padding_mode='reflect'):
    """Kuwahara filter for RGB images using luminance variance for region selection."""
    Hi, Wi, _ = image.shape
    radius = size // 2
    out = np.zeros((Hi, Wi, 3))

    image_padded = np.pad(
        image,
        ((radius, radius), (radius, radius), (0, 0)),
        mode=padding_mode,
    )

    for i in range(Hi):
        for j in range(Wi):
            patch = image_padded[i:i+size, j:j+size, :]
            intensity_patch = np.mean(patch, axis=2)

            intensity_regions = [
                intensity_patch[0:radius + 1, 0:radius + 1],
                intensity_patch[0:radius + 1, radius:size],
                intensity_patch[radius:size, 0:radius + 1],
                intensity_patch[radius:size, radius:size],
            ]
            color_regions = [
                patch[0:radius + 1, 0:radius + 1, :],
                patch[0:radius + 1, radius:size, :],
                patch[radius:size, 0:radius + 1, :],
                patch[radius:size, radius:size, :],
            ]

            variances = [np.var(region) for region in intensity_regions]
            best_idx = int(np.argmin(variances))
            out[i, j, :] = np.mean(color_regions[best_idx], axis=(0, 1))

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