from skimage import io, img_as_float, img_as_ubyte
from filters import engine, kernels
import os

def main():
    # 1. Load image
    img_path = 'assets/input/my_photo.jpg'
    image = img_as_float(io.imread(img_path))
    
    # 2. Select and apply filter
    kernel = kernels.get_emboss()
    art_image = engine.apply_filter(image, kernel)
    
    # 3. Save result
    if not os.path.exists('assets/output'):
        os.makedirs('assets/output')
    
    io.imsave('assets/output/embossed_art.png', img_as_ubyte(art_image.clip(0, 1)))
    print("Filter applied successfully!")

if __name__ == "__main__":
    main()