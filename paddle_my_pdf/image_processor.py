import cv2
import numpy as np

from .config import TARGET_DPI


class ImageProcessor:
    @staticmethod
    def get_skew_angle(img: np.ndarray) -> float:
        """Estimate the skew angle of a document image in degrees."""
        h, w = img.shape[:2]
        new_h = 800
        new_w = int(w * (new_h / h))
        small = cv2.resize(img, (new_w, new_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

    @staticmethod
    def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate *img* by *angle* degrees around its center."""
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    @staticmethod
    def detect_page_dpi(page, page_w: float, page_h: float) -> float:
        """Detect the effective DPI of the largest embedded image on *page*."""
        images = page.get_image_info()
        if not images:
            return float(TARGET_DPI)
        largest = max(images, key=lambda x: x["width"] * x["height"])
        img_w = largest["width"]
        img_h = largest["height"]
        dpi_x = img_w / (page_w / 72)
        dpi_y = img_h / (page_h / 72)
        detected = max(dpi_x, dpi_y)
        return min(detected, float(TARGET_DPI))
