#!/usr/bin/env python3
"""
Make a Chinese PDF searchable using PaddleOCR with lossy compression.

Usage:
    python pdf_searchable.py input.pdf output.pdf
    python pdf_searchable.py input.pdf output.pdf --model v5_normal
"""

import argparse
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image
from paddleocr import PaddleOCR
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


MODEL_CONFIGS = {
    "v5_lite": {
        "det": "PP-OCRv5_mobile_det",
        "rec": "PP-OCRv5_mobile_rec",
    },
    "v5_normal": {
        "det": "PP-OCRv5_server_det",
        "rec": "PP-OCRv5_server_rec",
    },
}


def render_pdf_to_images(pdf_path: str, dpi: int = 300):
    """Render PDF pages to PIL images."""
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append((img, float(pix.width), float(pix.height)))
    
    doc.close()
    return images


def ocr_page(image_path: str, ocr_engine):
    """Run OCR on an image and return results with bounding boxes."""
    result = ocr_engine.predict(image_path)
    if result and len(result) > 0:
        page_result = result[0]
        if 'rec_polys' in page_result and 'rec_texts' in page_result:
            polys = page_result['rec_polys']
            texts = page_result['rec_texts']
            scores = page_result.get('rec_scores', [1.0] * len(texts))
            
            ocr_results = []
            for i in range(len(texts)):
                bbox = polys[i].tolist()
                text = texts[i]
                conf = scores[i]
                ocr_results.append(([bbox, (text, conf)]))
            return ocr_results
    return []


def create_searchable_page(img, ocr_results, page_width, page_height):
    """Create a PDF page with image and invisible text overlay."""
    c = canvas.Canvas("", pagesize=(page_width, page_height))
    
    img_width, img_height = img.size
    scale_x = page_width / img_width
    scale_y = page_height / img_height
    
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG", quality=85)
    img_byte_arr.seek(0)
    
    c.drawImage(ImageReader(img_byte_arr), 0, 0, width=page_width, height=page_height)
    
    font_name = "Helvetica"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        
        font_path = os.path.expanduser("~/.fonts/NotoSansSC-Regular.ttf")
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('NotoSans', font_path))
            font_name = 'NotoSans'
    except Exception:
        pass
    
    for line in ocr_results:
        bbox = line[0]
        text = line[1][0]
        
        x0 = bbox[0][0] * scale_x
        y0 = page_height - bbox[0][1] * scale_y
        x1 = bbox[2][0] * scale_x
        y1 = page_height - bbox[2][1] * scale_y
        
        font_size = max(8, (y1 - y0) * 0.8)
        
        c.saveState()
        c.setFillColorRGB(0, 0, 0)
        c.translate(x0, y0)
        c.setFont(font_name, font_size)
        c.drawString(0, 0, text)
        c.restoreState()
    
    pdf_bytes = c.getpdfdata()
    return pdf_bytes


def compress_pdf(input_path: str, output_path: str, quality: int = 70):
    """Compress PDF using PyMuPDF with lossy image compression."""
    doc = fitz.open(input_path)
    
    doc.rewrite_images(dpi_target=150, quality=quality, lossy=True)
    
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Make Chinese PDF searchable with PaddleOCR")
    parser.add_argument("input_pdf", help="Input PDF path")
    parser.add_argument("output_pdf", help="Output PDF path")
    parser.add_argument(
        "--model",
        choices=["v5_lite", "v5_normal"],
        default="v5_lite",
        help="OCR model to use (default: v5_lite)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rendering PDF pages (default: 300)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=70,
        help="JPEG quality for compression (default: 70)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.input_pdf).exists():
        print(f"Error: Input file '{args.input_pdf}' not found")
        sys.exit(1)
    
    print(f"Processing: {args.input_pdf}")
    print(f"Model: {args.model}")
    print(f"DPI: {args.dpi}")
    print(f"Compression quality: {args.quality}")
    
    model_config = MODEL_CONFIGS[args.model]
    
    print("\nInitializing OCR engine...")
    ocr_engine = PaddleOCR(
        lang="ch",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_detection_model_name=model_config["det"],
        text_recognition_model_name=model_config["rec"],
    )
    
    print("Rendering PDF to images...")
    images = render_pdf_to_images(args.input_pdf, dpi=args.dpi)
    print(f"Found {len(images)} pages")
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        temp_pdf_path = tmp_file.name
    
    try:
        print("\nProcessing pages...")
        
        pdf_writer = fitz.open()
        
        for page_num, (img, width, height) in enumerate(images):
            print(f"  Processing page {page_num + 1}/{len(images)}...")
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name, format="JPEG", quality=95)
                ocr_results = ocr_page(f.name, ocr_engine)
                Path(f.name).unlink(missing_ok=True)
            
            page_pdf_data = create_searchable_page(img, ocr_results, width, height)
            page_pdf = fitz.open("pdf", page_pdf_data)
            pdf_writer.insert_pdf(page_pdf)
            page_pdf.close()
        
        pdf_writer.save(temp_pdf_path)
        pdf_writer.close()
        
        print("\nCompressing PDF...")
        compress_pdf(temp_pdf_path, args.output_pdf, quality=args.quality)
        
        original_size = Path(args.input_pdf).stat().st_size
        output_size = Path(args.output_pdf).stat().st_size
        ratio = (1 - output_size / original_size) * 100
        
        print(f"\nDone!")
        print(f"  Original:  {original_size / 1024 / 1024:.2f} MB")
        print(f"  Output:    {output_size / 1024 / 1024:.2f} MB")
        print(f"  Reduced by: {ratio:.1f}%")
        
    finally:
        Path(temp_pdf_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
