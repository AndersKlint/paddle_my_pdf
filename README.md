# Paddle My PDF

Make a Chinese PDF searchable using PaddleOCR. The tool adds an invisible text layer to scanned PDFs, enabling text search and selection.

## Features

- OCR for Chinese and English text in PDFs
- Optional page deskewing
- Rebuilds PDF with JPEG compression and invisible text layer
- Cross-platform (no Ghostscript required)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python paddle_my_pdf.py input.pdf output.pdf
```

### Options

- `--model {v4_lite,v4_normal,v5_lite,v5_normal,v5}` - OCR model (default: v4_lite)
- `--deskew` - Deskew pages before OCR
- `--skip-ocr` - Skip OCR (useful for deskewing only)

## Requirements

- paddleocr
- PyMuPDF
- numpy
- opencv-python
