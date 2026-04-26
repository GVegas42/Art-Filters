# Art Filters

Simple image filtering project using NumPy and scikit-image.

## Quick Start (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
python main.py
```

Run a single filter:

```powershell
python main.py --filter emboss
```

If your input photo is very large, control performance resizing:

```powershell
python main.py --max-dimension 1024
```

```powershell
python main.py --max-dimension 0
```

Tune the retro/pixel-art filter:

```powershell
python main.py --filter retro_pixel_art --retro-levels 3 --retro-dithering floyd_steinberg --retro-serpentine
```

```powershell
python main.py --filter retro_pixel_art --retro-levels 5 --retro-dithering ordered_bayer --retro-bayer-size 8
```

Available `--filter` values: `all`, `sharpen`, `emboss`, `outline`, `sobel_horizontal`, `sobel_vertical`, `scharr_horizontal`, `scharr_vertical`, `scharr_magnitude`, `prewitt_horizontal`, `prewitt_vertical`, `prewitt_magnitude`, `laplacian`, `laplacian_4`, `laplacian_8`, `gaussian_blur`, `log`, `gabor`, `median_oil_base`, `kuwahara_painterly`, `retro_pixel_art`

`scharr_magnitude` and `prewitt_magnitude` include a sketch boost pass (percentile contrast stretch + invert) for a pencil-like result.

Retro tuning flags (used by `retro_pixel_art`):

- `--retro-levels` (int, `>=2`): per-channel quantization levels, default `4`
- `--retro-dithering` (`none`, `ordered_bayer`, `floyd_steinberg`): default `floyd_steinberg`
- `--retro-bayer-size` (`2`, `4`, `8`): only relevant to `ordered_bayer`, default `4`
- `--retro-serpentine` / `--no-retro-serpentine`: serpentine Floyd-Steinberg scan toggle, default enabled

These flags also apply when you run `--filter all`; only `retro_pixel_art` consumes them.

Performance note:

- This project uses pure Python loops for filtering, which can be very slow on high-resolution images.
- By default, input images are auto-resized so their largest side is at most `1024` pixels (`--max-dimension 1024`).
- Use `--max-dimension 0` to disable auto-resize (full resolution, potentially much slower).
- While running, the app now prints progress like `[2/20] Applying 'emboss'...` so you can see it is still working.

## Input and Output Folders

The default paths used by [main.py](main.py) are:

- Input image: images/input/my_photo.jpg
- Output images: images/output/*_art.png

Expected structure:

```text
images/
	input/
		my_photo.jpg
	output/
```

First-run example:

1. Put a test image at images/input/my_photo.jpg
2. Run `python main.py` to generate every filter output, or `python main.py --filter emboss` to generate one.
3. Check the results in `images/output/` for one file per filter:
	- `sharpen_art.png`
	- `emboss_art.png`
	- `outline_art.png`
	- `sobel_horizontal_art.png`
	- `sobel_vertical_art.png`
	- `scharr_horizontal_art.png`
	- `scharr_vertical_art.png`
	- `scharr_magnitude_art.png`
	- `prewitt_horizontal_art.png`
	- `prewitt_vertical_art.png`
	- `prewitt_magnitude_art.png`
	- `laplacian_art.png`
	- `laplacian_4_art.png`
	- `laplacian_8_art.png`
	- `gaussian_blur_art.png`
	- `log_art.png`
	- `gabor_art.png`
	- `median_oil_base_art.png`
	- `kuwahara_painterly_art.png`
	- `retro_pixel_art_art.png`

## VS Code Interpreter

If imports such as NumPy or skimage are unresolved in VS Code, select the project interpreter:

- Path: .venv\Scripts\python.exe
- Command Palette: Python: Select Interpreter

This workspace already includes [.vscode/settings.json](.vscode/settings.json) configured to use that interpreter.

## Notes

- Package name: scikit-image
- Import name in code: skimage
