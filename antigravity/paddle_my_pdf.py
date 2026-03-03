#!/usr/bin/env python3
"""
paddle_my_pdf.py — Make a Chinese PDF searchable using PaddleOCR.

Renders each PDF page to an image, runs OCR to detect text and positions,
then reconstructs a new PDF with an invisible text layer over the page images.
Finally compresses the output with Ghostscript.

Usage:
    python paddle_my_pdf.py input.pdf output.pdf [--model {v4_lite,v4_normal,v5_lite,v5_normal,vl}]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# DPI for rendering PDF pages to images
RENDER_DPI = 200

# System CJK font for the invisible text layer
CJK_FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"

# Model configurations for standard PP-OCR models
MODEL_CONFIGS = {
    "v4_lite": {
        "ocr_version": "PP-OCRv4",
        "text_detection_model_name": "PP-OCRv4_mobile_det",
        "text_recognition_model_name": "PP-OCRv4_mobile_rec",
    },
    "v4_normal": {
        "ocr_version": "PP-OCRv4",
        "text_detection_model_name": "PP-OCRv4_server_det",
        "text_recognition_model_name": "PP-OCRv4_server_rec",
    },
    "v5_lite": {
        "ocr_version": "PP-OCRv5",
    },
    "v5_normal": {
        "ocr_version": "PP-OCRv5",
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "PP-OCRv5_server_rec",
    },
}


def init_ocr(model_key: str):
    """Initialize PaddleOCR or PaddleOCRVL with the chosen model configuration."""
    if model_key == "vl":
        from paddleocr import PaddleOCRVL
        return PaddleOCRVL(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    cfg = MODEL_CONFIGS[model_key]
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        **cfg,
    )


def extract_ocr_items(results, model_key: str):
    """
    Extract (text, score, poly) tuples from OCR results.
    Handles both standard PaddleOCR and PaddleOCR-VL output formats.
    """
    items = []
    for res in results:
        # Standard OCR: rec_texts + rec_polys
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])

        # VL model may use dt_polys instead of rec_polys
        if not polys:
            polys = res.get("dt_polys", [])
        if not scores:
            scores = res.get("dt_scores", [1.0] * len(texts))

        for text, score, poly in zip(texts, scores, polys):
            if score < 0.3 or not text.strip():
                continue
            items.append((text, score, np.array(poly, dtype=float)))
    return items


def build_searchable_pdf(input_pdf: str, raw_output: str, ocr_engine, model_key: str):
    """
    Build a searchable PDF by:
    1. Rendering each page as an image
    2. Running OCR on each page image
    3. Creating a new PDF with the image as background and invisible text overlay
    """
    src = fitz.open(input_pdf)
    dst = fitz.open()
    total_pages = len(src)

    # Load font once for TextWriter usage
    font = fitz.Font(fontfile=CJK_FONT_PATH)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for page_idx in range(total_pages):
            src_page = src[page_idx]
            page_w = src_page.rect.width
            page_h = src_page.rect.height
            print(f"  Processing page {page_idx + 1}/{total_pages}...")

            # Render to image
            zoom = RENDER_DPI / 72.0
            pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_w = pix.width
            img_h = pix.height

            img_path = str(tmpdir / f"page_{page_idx}.png")
            pix.save(img_path)

            # Create new page with same dimensions
            new_page = dst.new_page(width=page_w, height=page_h)

            # Insert page image as background
            new_page.insert_image(new_page.rect, filename=img_path)

            # Run OCR
            results = list(ocr_engine.predict(img_path))
            items = extract_ocr_items(results, model_key)

            # Log first few results for debugging
            for i, (text, score, _) in enumerate(items[:5]):
                print(f"    [{i+1}] score={score:.3f} text={text}")


            # Overlay invisible text with correct width scaling
            for text, score, poly in items:
                # Scale from image pixels to PDF points
                pts = poly.copy()
                pts[:, 0] *= page_w / img_w
                pts[:, 1] *= page_h / img_h

                x0 = float(pts[:, 0].min())
                y0 = float(pts[:, 1].min())
                x1 = float(pts[:, 0].max())
                y1 = float(pts[:, 1].max())
                box_w = x1 - x0
                box_h = y1 - y0

                # Small font size for correct selection height
                font_size = max(box_h * 0.35, 4)
                
                # Measure actual text width to calculate horizontal scale
                actual_w = font.text_length(text, fontsize=font_size)
                h_scale = box_w / actual_w if actual_w > 0 else 1.0

                # Insert invisible text (render_mode=3 = invisible)
                # Use morph for horizontal scaling to match box width
                origin = fitz.Point(x0, y1 - box_h * 0.1)
                new_page.insert_text(
                    origin,
                    text,
                    fontsize=font_size,
                    fontname="NotoSansCJK",
                    fontfile=CJK_FONT_PATH,
                    morph=(origin, fitz.Matrix(h_scale, 1)),
                    render_mode=3,
                )

            print(f"    Found {len(items)} text regions")

        src.close()
        dst.save(raw_output)
        dst.close()


def compress_pdf(input_path: str, output_path: str):
    """Compress using Ghostscript with lossy JPEG compression."""
    gs_cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dPDFSETTINGS=/ebook",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-sOutputFile={output_path}",
        input_path,
    ]
    try:
        subprocess.run(gs_cmd, check=True)
    except FileNotFoundError:
        print(
            "WARNING: ghostscript (gs) not found. Skipping compression. "
            "Install with: sudo apt install ghostscript",
            file=sys.stderr,
        )
        Path(output_path).write_bytes(Path(input_path).read_bytes())


def main():
    parser = argparse.ArgumentParser(
        description="Make a Chinese PDF searchable using PaddleOCR"
    )
    parser.add_argument("input", help="Path to input PDF")
    parser.add_argument("output", help="Path for output searchable PDF")
    parser.add_argument(
        "--model",
        choices=["v4_lite", "v4_normal", "v5_lite", "v5_normal", "vl"],
        default="v5_lite",
        help="OCR model variant (default: v5_lite)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")

    print("\n[1/3] Initializing PaddleOCR...")
    ocr = init_ocr(args.model)

    print("[2/3] Running OCR and building searchable PDF...")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        raw_path = tmp.name

    build_searchable_pdf(str(input_path), raw_path, ocr, args.model)

    print("[3/3] Compressing output PDF...")
    compress_pdf(raw_path, args.output)
    Path(raw_path).unlink(missing_ok=True)

    in_size = input_path.stat().st_size / 1024
    out_size = Path(args.output).stat().st_size / 1024
    print(f"\nDone! {in_size:.0f} KB → {out_size:.0f} KB")


if __name__ == "__main__":
    main()
