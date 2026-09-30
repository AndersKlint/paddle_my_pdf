# Paddle-My-PDF

Make Chinese PDFs searchable using PaddleOCR.

This tool deskews PDF pages, runs PaddleOCR to recognize text, and rebuilds the PDF with a searchable (invisible) text layer. It also uses Ghostscript for optimized compression.

## Features
- **Chinese OCR**: Optimized for Chinese language using PaddleOCR.
- **Deskewing**: Automatically corrects skewed scans.
- **Searchable Text**: Adds an invisible text layer over the original images.
- **Optimized Size**: Uses Ghostscript to compress the final PDF.
- **Parallel Processing**: Uses multi-threading for faster performance.

## Installation

### Prerequisites
- Python 3.8+
- [Ghostscript](https://www.ghostscript.com/) (optional, for compression)
- [libGL](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html) (for OpenCV)

### Install from source
```bash
git clone https://github.com/AndersKlint/paddle_my_pdf.git
cd paddle_my_pdf
pip install .
```

## Usage

After installation, you can use the `paddle-my-pdf` command:

```bash
paddle-my-pdf input.pdf output.pdf --model v5_lite --deskew
```

Alternatively, run as a module:
```bash
python -m paddle_my_pdf input.pdf output.pdf --threads 4
```

### Arguments
- `input`: Path to the input PDF.
- `output`: Path to the output PDF.
- `--model`: PaddleOCR model to use. Default is `v5_lite`.

  | Model | PP-OCR version | Detection | Recognition |
  |-------|----------------|-----------|-------------|
  | `v4_lite` | PP-OCRv4 | mobile | mobile |
  | `v4_normal` | PP-OCRv4 | server | server |
  | `v5_lite` | PP-OCRv5 | mobile | server |
  | `v5_normal` | PP-OCRv5 | server | server |
  | `v6_tiny` | PP-OCRv6 | tiny | tiny |
  | `v6_small` | PP-OCRv6 | small | small |
  | `v6_medium` | PP-OCRv6 | medium | medium |
  | `vl` | PaddleOCR-VL | — | — |

  PP-OCRv6 tiers differ from v4/v5: there is no "lite/normal" split, only
  `tiny` (fastest, 1.5M params), `small`, and `medium` (most accurate, 34.5M params).
  The v6 models cover Simplified Chinese, Traditional Chinese, English, Japanese
  and 46 Latin-script languages in a single recognition model — `v6_tiny` excludes
  Japanese.
- `--threads`: Number of threads for parallel page processing. Default is 4.
- `--deskew`: Enable automatic deskewing of pages.
- `--skip-ocr`: Skip OCR and only perform Ghostscript compression.

## Development

Run tests:
```bash
python tests/test_slow.py
```
