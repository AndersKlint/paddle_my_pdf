#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from .config import AppConfig
from .orchestrator import PDFOCROrchestrator

def main():
    parser = argparse.ArgumentParser(description="Make a Chinese PDF searchable using PaddleOCR (Modular Version)")

    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("output", help="Output PDF file")

    parser.add_argument(
        "--model",
        choices=["v4_lite", "v4_normal", "v5_lite", "v5_normal", "vl"],
        default="v5_lite",
        help="PaddleOCR model to use"
    )

    parser.add_argument("--threads", type=int, default=4, help="Number of threads for processing")
    parser.add_argument("--deskew", action="store_true", help="Attempt to deskew pages")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR and just compress")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    config = AppConfig(
        input_path=str(input_path.absolute()),
        output_path=args.output,
        model=args.model,
        threads=args.threads,
        deskew=args.deskew,
        skip_ocr=args.skip_ocr
    )

    orchestrator = PDFOCROrchestrator(config)
    
    try:
        orchestrator.run()
        
        in_size = input_path.stat().st_size / 1024
        out_size = Path(args.output).stat().st_size / 1024
        print(f"Done! {in_size:.0f} KB → {out_size:.0f} KB")
        
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
