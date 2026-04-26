from skimage import io, img_as_float, img_as_ubyte
from filters import engine, kernels
import os

def main():
    # 1. Load image
    img_path = os.path.join('images', 'input', 'my_photo.jpg')
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Input image not found at '{img_path}'. Add an image with that name or update the path in main.py."
        )
    image = img_as_float(io.imread(img_path))
    
    # 2. Select and apply filter
    kernel = kernels.get_emboss()
    art_image = engine.apply_filter(image, kernel)
    
    # 3. Save result
    output_dir = os.path.join('images', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'embossed_art.png')
    io.imsave(output_path, img_as_ubyte(art_image.clip(0, 1)))
    print("Filter applied successfully!")

if __name__ == "__main__":
    main()