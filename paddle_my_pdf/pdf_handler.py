import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

import fitz

from .config import JPEG_QUALITY, CJK_FONT_PATH
from .text_layer import TextOp


class PDFHandler:
    @staticmethod
    def compress(input_path: str, output_path: str) -> None:
        """Compress a PDF using Ghostscript, falling back to a raw copy."""
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
            print(
                "WARNING: ghostscript not found. Skipping compression.",
                file=sys.stderr,
            )
            Path(output_path).write_bytes(Path(input_path).read_bytes())

    @staticmethod
    def create_searchable_pdf(
        page_results: List[Tuple[str, List[TextOp]]], output_pdf: str
    ) -> None:
        """Merge per-page PDFs and overlay the invisible text layer."""
        dst_doc = fitz.open()

        for page_pdf, text_ops in page_results:
            with fitz.open(page_pdf) as pg:
                dst_doc.insert_pdf(pg)

            page = dst_doc[-1]

            for op in text_ops:
                try:
                    page.insert_text(
                        op.origin,
                        op.text,
                        fontsize=op.font_size,
                        fontname="NotoSansCJK",
                        fontfile=CJK_FONT_PATH,
                        morph=(op.origin, fitz.Matrix(op.h_scale, 1)),
                        render_mode=3,
                    )
                except Exception as exc:
                    print(
                        f"WARNING: skipped text '{op.text[:30]}': {exc}",
                        file=sys.stderr,
                    )

        dst_doc.save(
            output_pdf, garbage=4, clean=True, deflate=True, deflate_images=True
        )
        dst_doc.close()

        # Clean up temporary page PDFs
        for page_pdf, _ in page_results:
            if os.path.exists(page_pdf):
                os.remove(page_pdf)
