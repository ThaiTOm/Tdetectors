import numpy as np
import supervision as sv
from paddleocr import PaddleOCR
from paddleocr import TextRecognition


def crop_license_plates(frame: np.ndarray, lp_detections: sv.Detections) -> list:
    """
    Crops license plates from the frame based on detection bounding boxes.

    Args:
        frame (np.ndarray): The full video frame.
        lp_detections (sv.Detections): The license plate detections.

    Returns:
        list: A list of cropped license plate images (as np.ndarray).
    """
    cropped_images = []
    for xyxy in lp_detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        # Crop the image using numpy slicing
        cropped_image = frame[y1:y2, x1:x2]
        # Safety check for valid crop size
        if cropped_image.size > 0:
            cropped_images.append(cropped_image)
    return cropped_images

def run_ocr_on_plates(paddle_ocr: PaddleOCR, cropped_plates: list) -> list:
    """
    Performs OCR on a list of cropped license plate images.

    Args:
        paddle_ocr (PaddleOCR): The initialized PaddleOCR instance.
        cropped_plates (list): A list of cropped license plate images.

    Returns:
        list: A list of recognized text strings.
    """
    ocr_texts = []
    for crop in cropped_plates:
        # Perform OCR
        result = paddle_ocr.predict(crop)
        # Process OCR result
        plate_text = result[0]["rec_text"]
        if len(plate_text) > 4 and result[0]["rec_score"] > 0.9:
            ocr_texts.append(plate_text)
        else:
            ocr_texts.append("")
    return ocr_texts

if __name__ == "__main__":
    # Example usage
    ocr = TextRecognition()
    value = ocr.predict("Screenshot 2025-07-24 225503.png")
    print(value)
