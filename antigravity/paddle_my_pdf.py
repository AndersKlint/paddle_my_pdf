#!/usr/bin/env python3
"""
paddle_my_pdf.py — Make a Chinese PDF searchable using PaddleOCR.

Deskews pages if requested.
Rebuilds PDF with JPEG compression and invisible text layer.
Fully cross-platform. No Ghostscript needed.

Usage:
    python paddle_my_pdf.py input.pdf output.pdf
        [--model {v4_lite,v4_normal,v5_lite,v5_normal,vl}]
        [--deskew]
        [--skip-ocr]
"""

import argparse
import sys
import subprocess
import tempfile
import os
from pathlib import Path

import cv2

import numpy as np
import fitz  # PyMuPDF
import threading
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR

# DPI controls
# DPI and Resolution controls
DEFAULT_DPI = 150
MAX_DPI = 200
MAX_PIXEL_SIDE = 3000  # Cap resolution to prevent massive memory usage
JPEG_QUALITY = 80

CJK_FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"

MODEL_CONFIGS = {
    "v4_lite": {"ocr_version": "PP-OCRv4",
                "text_detection_model_name": "PP-OCRv4_mobile_det",
                "text_recognition_model_name": "PP-OCRv4_mobile_rec"},
    "v4_normal": {"ocr_version": "PP-OCRv4",
                  "text_detection_model_name": "PP-OCRv4_server_det",
                  "text_recognition_model_name": "PP-OCRv4_server_rec"},
    "v5_lite": {"ocr_version": "PP-OCRv5"},
    "v5_normal": {"ocr_version": "PP-OCRv5",
                  "text_detection_model_name": "PP-OCRv5_server_det",
                  "text_recognition_model_name": "PP-OCRv5_server_rec"},
}

ocr_lock = threading.Lock()


def init_ocr(model_key: str, deskew: bool = False):
    if model_key == "vl":
        from paddleocr import PaddleOCRVL
        return PaddleOCRVL(
            use_doc_orientation_classify=deskew,
            use_doc_unwarping=False,
        )
    cfg = MODEL_CONFIGS[model_key].copy()
    return PaddleOCR(
        use_doc_orientation_classify=deskew,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        **cfg,
    )


def extract_ocr_items(results):
    items = []
    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])
        if not polys:
            polys = res.get("dt_polys", [])
        if not scores:
            scores = res.get("dt_scores", [1.0] * len(texts))
        for text, score, poly in zip(texts, scores, polys):
            if score < 0.3 or not text.strip():
                continue
            items.append((text, np.array(poly, dtype=float)))
    return items


# ---- DESKEW ----

def get_skew_angle(img):
    h, w = img.shape[:2]
    new_h = 800
    new_w = int(w * (new_h / h))
    small = cv2.resize(img, (new_w, new_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        (rw, rh) = rect[1]
        if rw < rh:
            angle -= 90
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ---- MAIN PROCESSING ----

def process_page(page_idx, input_pdf, tmp_dir, ocr_engine, deskew):
    """Process a single page: Render -> Deskew -> OCR -> 1-page PDF."""
    src_doc = fitz.open(input_pdf)
    page = src_doc[page_idx]
    page_w = page.rect.width
    page_h = page.rect.height

    # 1. Render to image with resolution intelligence
    # Inspect page for existing images to determine "native" resolution
    images = page.get_image_info()
    if images:
        # Find the largest image by area
        largest_img = max(images, key=lambda x: x["width"] * x["height"])
        img_w_raw = largest_img["width"]
        img_h_raw = largest_img["height"]
        # Determine zoom to match this resolution
        zoom = min(img_w_raw / page_w, img_h_raw / page_h)
    else:
        # Fallback to DEFAULT_DPI if no images found
        zoom = DEFAULT_DPI / 72.0
    
    # Safety check: ensure zoom doesn't exceed our pixel limit
    if page_w * zoom > MAX_PIXEL_SIDE or page_h * zoom > MAX_PIXEL_SIDE:
        zoom = min(MAX_PIXEL_SIDE / page_w, MAX_PIXEL_SIDE / page_h)

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # 2. Deskew
    if deskew:
        angle = get_skew_angle(img_bgr)
        if abs(angle) > 0.1:
            img_bgr = rotate_image(img_bgr, angle)
            print(f"  Page {page_idx+1}: Deskewed by {-angle:.2f}°")

    img_h, img_w = img_bgr.shape[:2]

    # 3. Write temp JPEG
    tmp_img_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.jpg")
    cv2.imwrite(tmp_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

    # 4. OCR (Thread-locked)
    with ocr_lock:
        results = list(ocr_engine.predict(tmp_img_path))
    items = extract_ocr_items(results)
    print(f"  Page {page_idx+1}: {len(items)} text regions")

    # 5. Build 1-page PDF
    page_pdf_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.pdf")
    pg_doc = fitz.open()
    # Adjust page height if aspect ratio changed due to deskew/rotation
    final_page_h = (img_h / img_w) * page_w
    new_page = pg_doc.new_page(width=page_w, height=final_page_h)
    new_page.insert_image(new_page.rect, filename=tmp_img_path)
    
    font = fitz.Font(fontfile=CJK_FONT_PATH)
    for text, poly in items:
        pts = poly.copy()
        pts[:, 0] *= page_w / img_w
        pts[:, 1] *= final_page_h / img_h

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

    pg_doc.save(page_pdf_path, garbage=4, clean=True, deflate=True)
    pg_doc.close()
    src_doc.close()
    os.remove(tmp_img_path)
    
    return page_pdf_path


def build_searchable_pdf(input_pdf, output_pdf, ocr_engine, threads, deskew):
    src_doc = fitz.open(input_pdf)
    total_pages = len(src_doc)
    src_doc.close()
    
    print(f"Processing {total_pages} pages with {threads} threads...")

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(process_page, i, input_pdf, tmp_dir_name, ocr_engine, deskew)
                for i in range(total_pages)
            ]
            page_pdfs = [f.result() for f in futures]

        print("Merging pages...")
        dst_doc = fitz.open()
        for p_pdf in sorted(page_pdfs):
            with fitz.open(p_pdf) as pg:
                dst_doc.insert_pdf(pg)
        
        dst_doc.save(output_pdf, garbage=4, clean=True, deflate=True)
        dst_doc.close()


def compress_pdf(input_path: str, output_path: str):
    """Compress using Ghostscript to subset fonts and apply lossy JPEG compression."""
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
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--model",
        choices=["v4_lite", "v4_normal", "v5_lite", "v5_normal", "vl"],
        default="v5_lite",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print("Input file not found.", file=sys.stderr)
        sys.exit(1)

    if args.skip_ocr:
        compress_pdf(str(input_path), args.output)
        return

    print("Initializing PaddleOCR...")
    ocr = init_ocr(args.model, args.deskew)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        raw_path = tmp.name

    build_searchable_pdf(str(input_path), raw_path, ocr, args.threads, args.deskew)

    print("Compressing output PDF...")
    compress_pdf(raw_path, args.output)
    Path(raw_path).unlink(missing_ok=True)

    in_size = input_path.stat().st_size / 1024
    out_size = Path(args.output).stat().st_size / 1024
    print(f"Done! {in_size:.0f} KB → {out_size:.0f} KB")


if __name__ == "__main__":
    main()