#!/usr/bin/env python3
"""
paddle_my_pdf.py — Make a Chinese PDF searchable using PaddleOCR.

Renders each PDF page to an image, runs OCR to detect text and positions,
then reconstructs a new PDF with an invisible text layer over the page images.
Finally compresses the output with Ghostscript.

Usage:
    python paddle_my_pdf.py input.pdf output.pdf [--model {v4_lite,v4_normal,v5_lite,v5_normal,vl}] [--threads N] [--deskew]
"""

import argparse
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
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

# Global lock for the OCR engine if needed
ocr_lock = threading.Lock()


def init_ocr(model_key: str, deskew: bool = False):
    """Initialize PaddleOCR or PaddleOCRVL with the chosen model configuration."""
    if model_key == "vl":
        from paddleocr import PaddleOCRVL
        return PaddleOCRVL(
            use_doc_orientation_classify=deskew,
            use_doc_unwarping=False,  # We handle deskewing manually
        )
    cfg = MODEL_CONFIGS[model_key].copy()
    return PaddleOCR(
        use_doc_orientation_classify=deskew,
        use_doc_unwarping=False,  # We handle deskewing manually
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


def get_skew_angle(img):
    """
    Detect skew angle of a document image based on text line orientation.
    Returns the angle in degrees to rotate the image for deskewing.
    """
    h, w = img.shape[:2]
    # Resize for faster processing
    new_h = 800
    new_w = int(w * (new_h / h))
    small = cv2.resize(img, (new_w, new_h))
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Binary inverse (text becomes white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Dilate to merge characters into horizontal blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        
        # Adjust angle based on rectangle orientation (OpenCV 4.5+ logic)
        (rw, rh) = rect[1]
        if rw < rh:
            angle = angle - 90
            
        if -45 < angle < 45:
            angles.append(angle)
            
    if not angles:
        return 0.0
    
    return np.median(angles)


def rotate_image(img, angle):
    """Rotate image by the given angle (degrees)."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use white border for PDF pages
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated


def process_page(page_idx, input_pdf_path, tmp_dir, ocr_engine, model_key, deskew):
    """
    Process a single page: Render -> Deskew -> OCR -> Reconstruct as 1-page PDF.
    Returns the path to the individual page PDF.
    """
    input_pdf_path = str(input_pdf_path)
    tmp_dir = Path(tmp_dir)
    page_pdf_path = tmp_dir / f"page_{page_idx:05d}.pdf"
    
    with fitz.open(input_pdf_path) as doc:
        page = doc[page_idx]
        page_w = page.rect.width
        page_h = page.rect.height
        
        # 1. Render to image
        zoom = RENDER_DPI / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # 2. Deskew if requested
        bg_img_path = str(tmp_dir / f"page_proc_{page_idx:05d}.png")
        if deskew:
            angle = get_skew_angle(img_bgr)
            if abs(angle) > 0.1:
                img_bgr = rotate_image(img_bgr, angle)
                print(f"  Page {page_idx + 1}: Deskewed by {-angle:.2f} degrees")
        
        cv2.imwrite(bg_img_path, img_bgr)
        img_h, img_w = img_bgr.shape[:2]

        # 3. Run OCR
        with ocr_lock:
            results = list(ocr_engine.predict(bg_img_path))
        
        items = extract_ocr_items(results)
        print(f"  Page {page_idx + 1}: Found {len(items)} text regions")
            
        # 4. Build new 1-page PDF
        dst_doc = fitz.open()
        # Maintain aspect ratio of potentially deskewed image
        page_h = (img_h / img_w) * page_w
        new_page = dst_doc.new_page(width=page_w, height=page_h)
        new_page.insert_image(new_page.rect, filename=bg_img_path)
        
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
            
            font_size = max(box_h * 0.35, 4)
            actual_w = font.text_length(text, fontsize=font_size)
            h_scale = box_w / actual_w if actual_w > 0 else 1.0
            
            origin = fitz.Point(x0, y1 - box_h * 0.15)
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


def build_searchable_pdf(input_pdf: str, raw_output: str, ocr_engine, model_key: str, threads: int, deskew: bool):
    """
    Build a searchable PDF using multithreading.
    """
    doc = fitz.open(input_pdf)
    total_pages = len(doc)
    doc.close()
    
    print(f"Processing {total_pages} pages using {threads} threads...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(process_page, i, input_pdf, tmp_dir, ocr_engine, model_key, deskew)
                for i in range(total_pages)
            ]
            page_pdfs = [f.result() for f in futures]
            
        print("\nMerging pages...")
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
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Enable automatic deskewing and unwarping of pages",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    print(f"Deskew: {args.deskew}")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")

    print("\n[1/3] Initializing PaddleOCR...")
    ocr = init_ocr(args.model, args.deskew)

    print("\n[2/3] Running OCR and building searchable PDF...")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        raw_path = tmp.name

    build_searchable_pdf(str(input_path), raw_path, ocr, args.model, args.threads, args.deskew)

    print("\n[3/3] Compressing output PDF...")
    compress_pdf(raw_path, args.output)
    Path(raw_path).unlink(missing_ok=True)

    in_size = input_path.stat().st_size / 1024
    out_size = Path(args.output).stat().st_size / 1024
    print(f"\nDone! {in_size:.0f} KB → {out_size:.0f} KB")


if __name__ == "__main__":
    main()
