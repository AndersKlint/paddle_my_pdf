from dataclasses import dataclass
from typing import Any, Dict

TARGET_DPI = 300
JPEG_QUALITY = 85
MIN_OCR_CONFIDENCE = 0.3
CJK_FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
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


@dataclass
class AppConfig:
    input_path: str
    output_path: str
    model: str = "v5_lite"
    threads: int = 4
    deskew: bool = False
    skip_ocr: bool = False
