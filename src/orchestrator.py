import os
import tempfile
import cv2
import numpy as np
import fitz
from concurrent.futures import ThreadPoolExecutor
from .config import AppConfig, JPEG_QUALITY, CJK_FONT_PATH
from .ocr_manager import OCRManager
from .image_processor import ImageProcessor
from .pdf_handler import PDFHandler

class PDFOCROrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr_manager = OCRManager(config.model, config.deskew)

    def run(self):
        if self.config.skip_ocr:
            print("Skipping OCR as requested. Compressing input PDF...")
            PDFHandler.compress(self.config.input_path, self.config.output_path)
            return

        print(f"Initializing OCR with model: {self.config.model}")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            raw_path = tmp.name

        try:
            self.build_searchable_pdf(raw_path)
            print("Compressing output PDF...")
            PDFHandler.compress(raw_path, self.config.output_path)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

    def build_searchable_pdf(self, output_pdf):
        src_doc = fitz.open(self.config.input_path)
        total_pages = len(src_doc)
        src_doc.close()

        print(f"Processing {total_pages} pages with {self.config.threads} threads...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
                futures = [
                    executor.submit(self.process_page, i, tmp_dir)
                    for i in range(total_pages)
                ]
                page_results = [f.result() for f in futures]

            print("Merging pages and adding text layer...")
            PDFHandler.create_searchable_pdf(page_results, output_pdf)

    def process_page(self, page_idx, tmp_dir):
        src_doc = fitz.open(self.config.input_path)
        page = src_doc[page_idx]

        page_w = page.rect.width
        page_h = page.rect.height

        dpi = ImageProcessor.detect_page_dpi(page, page_w, page_h)
        zoom = dpi / 72.0

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        if self.config.deskew:
            angle = ImageProcessor.get_skew_angle(img_bgr)
            if abs(angle) > 0.1:
                img_bgr = ImageProcessor.rotate_image(img_bgr, angle)
                print(f"  Page {page_idx+1}: Deskewed by {-angle:.2f}°")

        img_h, img_w = img_bgr.shape[:2]
        
        # OCR
        items = self.ocr_manager.predict(img_bgr)
        print(f"  Page {page_idx+1}: {len(items)} text regions ({dpi:.0f} DPI)")

        # Save temporary image
        tmp_img_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.jpg")
        cv2.imwrite(tmp_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        # Create individual page PDF
        page_pdf_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.pdf")
        pg_doc = fitz.open()
        final_page_h = (img_h / img_w) * page_w
        new_page = pg_doc.new_page(width=page_w, height=final_page_h)
        new_page.insert_image(new_page.rect, filename=tmp_img_path)

        font = fitz.Font(fontfile=CJK_FONT_PATH)
        text_ops = []

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
            text_ops.append((origin, text, font_size, h_scale))

        pg_doc.save(page_pdf_path, garbage=4, clean=True, deflate=True, deflate_images=True)
        pg_doc.close()
        src_doc.close()
        os.remove(tmp_img_path)

        return page_pdf_path, text_ops
