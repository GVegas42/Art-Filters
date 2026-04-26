import argparse
import os
import time

import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.transform import resize
from filters import engine, kernels


def min_retro_levels(value):
    int_value = int(value)
    if int_value < 2:
        raise argparse.ArgumentTypeError("--retro-levels must be >= 2")
    return int_value


def parse_args():
    available_filters = list(kernels.get_all_filters())
    parser = argparse.ArgumentParser(
        description="Apply one art filter or every available filter to the input image."
    )
    parser.add_argument(
        "--filter",
        choices=["all", *available_filters],
        default="all",
        help="Filter to apply. Defaults to all filters.",
    )
    parser.add_argument(
        "--retro-levels",
        type=min_retro_levels,
        default=4,
        help="Color levels per channel for retro_pixel_art (>=2). Defaults to 4.",
    )
    parser.add_argument(
        "--retro-dithering",
        choices=["none", "ordered_bayer", "floyd_steinberg"],
        default="floyd_steinberg",
        help="Dithering mode for retro_pixel_art. Defaults to floyd_steinberg.",
    )
    parser.add_argument(
        "--retro-bayer-size",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Bayer matrix size for ordered_bayer dithering (2,4,8). Defaults to 4.",
    )
    parser.add_argument(
        "--retro-serpentine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable serpentine scan for Floyd-Steinberg dithering. Defaults to enabled.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help=(
            "Auto-resize input if its largest side exceeds this value to avoid long runtimes. "
            "Use 0 to disable. Defaults to 1024."
        ),
    )
    return parser.parse_args()


def prepare_output_image(image):
    min_value = np.min(image)
    max_value = np.max(image)

    if min_value < 0 or max_value > 1:
        if max_value == min_value:
            return np.zeros_like(image)
        return (image - min_value) / (max_value - min_value)

    return image.clip(0, 1)


def apply_cli_overrides(filter_name, filter_spec, args):
    if not isinstance(filter_spec, dict):
        return filter_spec

    if filter_name != "retro_pixel_art":
        return filter_spec

    if filter_spec.get("type") != "quantize_dither":
        return filter_spec

    updated_spec = dict(filter_spec)
    updated_spec["levels"] = args.retro_levels
    updated_spec["dithering"] = args.retro_dithering
    updated_spec["bayer_size"] = args.retro_bayer_size
    updated_spec["serpentine"] = args.retro_serpentine
    return updated_spec


def maybe_resize_for_performance(image, max_dimension):
    if max_dimension <= 0:
        return image

    height, width = image.shape[:2]
    largest_side = max(height, width)

    if largest_side <= max_dimension:
        return image

    scale = max_dimension / float(largest_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    if image.ndim == 3:
        output_shape = (new_height, new_width, image.shape[2])
    else:
        output_shape = (new_height, new_width)

    print(
        f"Auto-resizing input from {width}x{height} to {new_width}x{new_height} "
        f"(max side {max_dimension}) for performance."
    )
    return resize(image, output_shape, anti_aliasing=True, preserve_range=True)

def main():
    args = parse_args()
    available_filters = kernels.get_all_filters()

    # 1. Load image
    img_path = os.path.join('images', 'input', 'my_photo.jpg')
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Input image not found at '{img_path}'. Add an image with that name or update the path in main.py."
        )
    image = img_as_float(io.imread(img_path))
    image = maybe_resize_for_performance(image, args.max_dimension)
    
    # 2. Apply every available filter and save each result
    output_dir = os.path.join('images', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_filters = available_filters
    if args.filter != "all":
        selected_filters = {args.filter: available_filters[args.filter]}

    total_filters = len(selected_filters)
    for index, (filter_name, filter_spec) in enumerate(selected_filters.items(), start=1):
        print(f"[{index}/{total_filters}] Applying '{filter_name}'...")
        start_time = time.perf_counter()

        effective_spec = apply_cli_overrides(filter_name, filter_spec, args)
        art_image = engine.apply_art_filter(image, effective_spec)
        output_path = os.path.join(output_dir, f'{filter_name}_art.png')
        io.imsave(output_path, img_as_ubyte(prepare_output_image(art_image)))
        elapsed = time.perf_counter() - start_time
        print(f"Saved {output_path} ({elapsed:.2f}s)")

if __name__ == "__main__":
    main()