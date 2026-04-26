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

Available `--filter` values: `all`, `sharpen`, `emboss`, `outline`, `sobel_horizontal`, `sobel_vertical`, `laplacian`, `laplacian_4`, `laplacian_8`, `gaussian_blur`, `log`, `gabor`, `median_oil_base`

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
	- `laplacian_art.png`
	- `laplacian_4_art.png`
	- `laplacian_8_art.png`
	- `gaussian_blur_art.png`
	- `log_art.png`
	- `gabor_art.png`
	- `median_oil_base_art.png`

## VS Code Interpreter

If imports such as NumPy or skimage are unresolved in VS Code, select the project interpreter:

- Path: .venv\Scripts\python.exe
- Command Palette: Python: Select Interpreter

This workspace already includes [.vscode/settings.json](.vscode/settings.json) configured to use that interpreter.

## Notes

- Package name: scikit-image
- Import name in code: skimage
