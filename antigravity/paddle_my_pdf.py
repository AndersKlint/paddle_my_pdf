#!/usr/bin/env python3
"""
paddle_my_pdf.py — Make a Chinese PDF searchable using PaddleOCR.

Renders each PDF page to an image, runs OCR to detect text and positions,
then reconstructs a new PDF with an invisible text layer over the page images.
Finally compresses the output with Ghostscript.

Usage:
    python paddle_my_pdf.py input.pdf output.pdf [--model {v4_lite,v4_normal,v5_lite,v5_normal,vl}] [--threads N]
"""

import argparse
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
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

# Global lock for the OCR engine if needed (standard PaddleOCR is generally thread-safe for inference)
ocr_lock = threading.Lock()


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


def extract_ocr_items(results):
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


def process_page(page_idx, input_pdf_path, tmp_dir, ocr_engine, model_key):
    """
    Process a single page: Render -> OCR -> Reconstruct as 1-page PDF.
    Returns the path to the individual page PDF.
    """
    input_pdf_path = str(input_pdf_path)
    tmp_dir = Path(tmp_dir)
    page_pdf_path = tmp_dir / f"page_{page_idx:05d}.pdf"
    
    # Each thread opens its own PyMuPDF document for thread-safety
    with fitz.open(input_pdf_path) as doc:
        page = doc[page_idx]
        page_w = page.rect.width
        page_h = page.rect.height
        
        # Render to image
        zoom = RENDER_DPI / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_w = pix.width
        img_h = pix.height
        
        img_path = str(tmp_dir / f"page_{page_idx:05d}.png")
        pix.save(img_path)
        
        # Run OCR (with lock for safety, though Paddle often handles concurrency well internally)
        with ocr_lock:
            results = list(ocr_engine.predict(img_path))
        
        items = extract_ocr_items(results)
        
        # Print progress summary
        print(f"  Page {page_idx + 1}: Found {len(items)} text regions")
        if items:
            print(f"    Sample: {items[0][0]}")
            
        # Build new 1-page PDF
        dst_doc = fitz.open()
        new_page = dst_doc.new_page(width=page_w, height=page_h)
        new_page.insert_image(new_page.rect, filename=img_path)
        
        font = fitz.Font(fontfile=CJK_FONT_PATH)
        
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
            
            # Insert invisible text
            origin = fitz.Point(x0, y1 - box_h * 0.1)
            try:
                new_page.insert_text(
                    origin,
                    text,
                    fontsize=font_size,
                    fontname="NotoSansCJK",
                    fontfile=CJK_FONT_PATH,
                    morph=(origin, fitz.Matrix(h_scale, 1)),
                    render_mode=3,
                )
            except Exception:
                continue
                
        dst_doc.save(str(page_pdf_path))
        dst_doc.close()
        
    return page_pdf_path


def build_searchable_pdf(input_pdf: str, raw_output: str, ocr_engine, model_key: str, threads: int):
    """
    Build a searchable PDF using multithreading.
    """
    doc = fitz.open(input_pdf)
    total_pages = len(doc)
    doc.close()
    
    print(f"Processing {total_pages} pages using {threads} threads...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(process_page, i, input_pdf, tmp_dir, ocr_engine, model_key)
                for i in range(total_pages)
            ]
            page_pdfs = [f.result() for f in futures]
            
        print("\nMerging pages...")
        # Merge all single-page PDFs in order
        merged_doc = fitz.open()
        for p_pdf in sorted(page_pdfs):
            with fitz.open(str(p_pdf)) as page_doc:
                merged_doc.insert_pdf(page_doc)
        merged_doc.save(raw_output)
        merged_doc.close()


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
        description="Make a Chinese PDF searchable using PaddleOCR (parallelized)"
    )
    parser.add_argument("input", help="Path to input PDF")
    parser.add_argument("output", help="Path for output searchable PDF")
    parser.add_argument(
        "--model",
        choices=["v4_lite", "v4_normal", "v5_lite", "v5_normal", "vl"],
        default="v5_lite",
        help="OCR model variant (default: v5_lite)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for parallel page processing (default: 4)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")

    print("\n[1/3] Initializing PaddleOCR...")
    ocr = init_ocr(args.model)

    print("\n[2/3] Running OCR and building searchable PDF...")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        raw_path = tmp.name

    build_searchable_pdf(str(input_path), raw_path, ocr, args.model, args.threads)

    print("\n[3/3] Compressing output PDF...")
    compress_pdf(raw_path, args.output)
    Path(raw_path).unlink(missing_ok=True)

    in_size = input_path.stat().st_size / 1024
    out_size = Path(args.output).stat().st_size / 1024
    print(f"\nDone! {in_size:.0f} KB → {out_size:.0f} KB")


if __name__ == "__main__":
    main()
