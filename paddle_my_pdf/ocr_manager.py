import threading
from typing import List, Tuple

import numpy as np
from paddleocr import PaddleOCR

from .config import MODEL_CONFIGS, MIN_OCR_CONFIDENCE


class OCRManager:
    def __init__(self, model_key: str, deskew: bool = False) -> None:
        self.lock = threading.Lock()
        self.model_key = model_key
        self.deskew = deskew
        self.engine = self._init_ocr()

    def _init_ocr(self) -> PaddleOCR:
        if self.model_key == "vl":
            from paddleocr import PaddleOCRVL
            return PaddleOCRVL(
                use_doc_orientation_classify=self.deskew,
                use_doc_unwarping=False,
            )

        cfg = MODEL_CONFIGS[self.model_key].copy()
        return PaddleOCR(
            use_doc_orientation_classify=self.deskew,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **cfg,
        )

    def predict(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Run OCR on *image* and return ``(text, polygon)`` pairs."""
        with self.lock:
            results = list(self.engine.predict(image))
        return self._extract_ocr_items(results)

    @staticmethod
    def _extract_ocr_items(results) -> List[Tuple[str, np.ndarray]]:
        items: List[Tuple[str, np.ndarray]] = []
        for res in results:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", [])

            if not polys:
                polys = res.get("dt_polys", [])

            if not scores:
                scores = res.get("dt_scores", [1.0] * len(texts))

            for text, score, poly in zip(texts, scores, polys):
                if score < MIN_OCR_CONFIDENCE or not text.strip():
                    continue
                items.append((text, np.array(poly, dtype=float)))
        return items
