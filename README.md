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

## Input and Output Folders

The default paths used by [main.py](main.py) are:

- Input image: images/input/my_photo.jpg
- Output image: images/output/embossed_art.png

Expected structure:

```text
images/
	input/
		my_photo.jpg
	output/
```

First-run example:

1. Put a test image at images/input/my_photo.jpg
2. Run `python main.py`
3. Check the result at images/output/embossed_art.png

## VS Code Interpreter

If imports such as NumPy or skimage are unresolved in VS Code, select the project interpreter:

- Path: .venv\Scripts\python.exe
- Command Palette: Python: Select Interpreter

This workspace already includes [.vscode/settings.json](.vscode/settings.json) configured to use that interpreter.

## Notes

- Package name: scikit-image
- Import name in code: skimage
