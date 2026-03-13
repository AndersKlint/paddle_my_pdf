import os
import sys
import subprocess
from pathlib import Path
import fitz
from .config import JPEG_QUALITY, CJK_FONT_PATH

class PDFHandler:
    @staticmethod
    def compress(input_path: str, output_path: str):
        gs_cmd = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/printer",
            "-dColorConversionStrategy=/sRGB",
            "-dProcessColorModel=/DeviceRGB",
            "-dAutoFilterColorImages=true",
            "-dColorImageFilter=/DCTEncode",
            f"-dJPEGQuality={JPEG_QUALITY}",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            f"-sOutputFile={output_path}",
            input_path,
        ]

        try:
            subprocess.run(gs_cmd, check=True)
        except FileNotFoundError:
            print("WARNING: ghostscript not found. Skipping compression.", file=sys.stderr)
            Path(output_path).write_bytes(Path(input_path).read_bytes())

    @staticmethod
    def create_searchable_pdf(page_results, output_pdf):
        dst_doc = fitz.open()

        for p_pdf, text_ops in page_results:
            with fitz.open(p_pdf) as pg:
                dst_doc.insert_pdf(pg)

            new_page = dst_doc[-1]
            
            # Re-obtain page dimensions from the insert
            page_w = new_page.rect.width
            
            for origin, text, font_size, h_scale in text_ops:
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

        dst_doc.save(output_pdf, garbage=4, clean=True, deflate=True, deflate_images=True)
        dst_doc.close()
        
        # Cleanup temporary page PDFs
        for p_pdf, _ in page_results:
            if os.path.exists(p_pdf):
                os.remove(p_pdf)
