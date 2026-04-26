import argparse
import os

import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from filters import engine, kernels


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
    return parser.parse_args()


def prepare_output_image(image):
    min_value = np.min(image)
    max_value = np.max(image)

    if min_value < 0 or max_value > 1:
        if max_value == min_value:
            return np.zeros_like(image)
        return (image - min_value) / (max_value - min_value)

    return image.clip(0, 1)

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
    
    # 2. Apply every available filter and save each result
    output_dir = os.path.join('images', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_filters = available_filters
    if args.filter != "all":
        selected_filters = {args.filter: available_filters[args.filter]}

    for filter_name, filter_spec in selected_filters.items():
        art_image = engine.apply_art_filter(image, filter_spec)
        output_path = os.path.join(output_dir, f'{filter_name}_art.png')
        io.imsave(output_path, img_as_ubyte(prepare_output_image(art_image)))
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()