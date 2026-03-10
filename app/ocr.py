"""OCR pipeline for image-based medical records."""
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter, ImageOps

from app.utils import clean_text


class OCRService:
    """Lightweight OCR service built on Tesseract."""

    def extract_text(self, image_path: Path) -> str:
        """Read text from an image file."""
        image = Image.open(image_path)
        processed = self._preprocess(image)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        return clean_text(text)

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        """Apply simple preprocessing to improve OCR quality."""
        grayscale = ImageOps.grayscale(image)
        autocontrast = ImageOps.autocontrast(grayscale)
        sharpened = autocontrast.filter(ImageFilter.SHARPEN)
        return sharpened
