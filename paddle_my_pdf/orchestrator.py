import os
import tempfile
from typing import List, Tuple

import cv2
import fitz
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .config import AppConfig, JPEG_QUALITY
from .image_processor import ImageProcessor
from .ocr_manager import OCRManager
from .pdf_handler import PDFHandler
from .text_layer import TextOp, compute_text_ops


class PDFOCROrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr_manager = OCRManager(config.model, config.deskew)

    def run(self) -> None:
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

    def build_searchable_pdf(self, output_pdf: str) -> None:
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

    def process_page(
        self, page_idx: int, tmp_dir: str
    ) -> Tuple[str, List[TextOp]]:
        src_doc = fitz.open(self.config.input_path)
        page = src_doc[page_idx]
        page_w = page.rect.width
        page_h = page.rect.height

        # Render page at detected DPI
        dpi = ImageProcessor.detect_page_dpi(page, page_w, page_h)
        img_bgr = self._render_page(page, dpi)
        src_doc.close()

        # Optional deskewing
        if self.config.deskew:
            img_bgr = self._maybe_deskew(img_bgr, page_idx)

        img_h, img_w = img_bgr.shape[:2]

        # OCR
        items = self.ocr_manager.predict(img_bgr)
        print(f"  Page {page_idx + 1}: {len(items)} text regions ({dpi:.0f} DPI)")

        # Build per-page PDF containing only the background image
        page_pdf_path, final_page_h = self._build_page_pdf(
            img_bgr, img_w, img_h, page_w, page_idx, tmp_dir
        )

        # Compute invisible text layer operations
        text_ops = compute_text_ops(items, page_w, final_page_h, img_w, img_h)

        return page_pdf_path, text_ops

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_page(page: fitz.Page, dpi: float) -> np.ndarray:
        """Render a PDF page to a BGR numpy array at the given DPI."""
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )
        if pix.n == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    def _maybe_deskew(self, img_bgr: np.ndarray, page_idx: int) -> np.ndarray:
        """Deskew the image if the detected skew exceeds 0.1 degrees."""
        angle = ImageProcessor.get_skew_angle(img_bgr)
        if abs(angle) > 0.1:
            img_bgr = ImageProcessor.rotate_image(img_bgr, angle)
            print(f"  Page {page_idx + 1}: Deskewed by {-angle:.2f} deg")
        return img_bgr

    @staticmethod
    def _build_page_pdf(
        img_bgr: np.ndarray,
        img_w: int,
        img_h: int,
        page_w: float,
        page_idx: int,
        tmp_dir: str,
    ) -> Tuple[str, float]:
        """Write the rasterised page image into a single-page PDF.

        Returns ``(path_to_page_pdf, page_height_in_points)``.
        """
        tmp_img_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.jpg")
        cv2.imwrite(
            tmp_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        page_pdf_path = os.path.join(tmp_dir, f"page_{page_idx:05d}.pdf")
        final_page_h = (img_h / img_w) * page_w

        pg_doc = fitz.open()
        new_page = pg_doc.new_page(width=page_w, height=final_page_h)
        new_page.insert_image(new_page.rect, filename=tmp_img_path)
        pg_doc.save(
            page_pdf_path, garbage=4, clean=True, deflate=True, deflate_images=True
        )
        pg_doc.close()
        os.remove(tmp_img_path)

        return page_pdf_path, final_page_h
