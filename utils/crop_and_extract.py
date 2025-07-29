import numpy as np
import supervision as sv
from paddleocr import PaddleOCR
from paddleocr import TextRecognition
import re

def delete_non_alphanumeric(text):
    """
    Deletes all characters from a string that are not letters or numbers.

    Args:
        text: The input string.

    Returns:
        A new string containing only alphanumeric characters.
    """
    # The regex [^a-zA-Z0-9] matches any character that is NOT (^)
    # a lowercase letter (a-z), an uppercase letter (A-Z), or a digit (0-9).
    # The re.sub() function replaces all occurrences of the pattern with an empty string.
    return re.sub(r'[^a-zA-Z0-9]', '', text)

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

def run_ocr_on_plates(paddle_ocr, cropped_plates: list) -> list:
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
        result = paddle_ocr.predict(
            input=crop)
        mn = min(result[0]["rec_scores"]) if result[0]["rec_scores"] else 0.0
        if len(result[0]["rec_texts"]) > 1:
            ocr_texts.append(("".join(result[0]["rec_texts"]), mn))
        else:
            ocr_texts.append(("", 0.0))
    return ocr_texts

def LicensePlateReader(model, cropped_plates: list):
    ocr_texts = []
    for crop in cropped_plates:
        # Perform OCR
        txt, score = model.run(crop, return_confidence=True)
        score = np.mean(score)
        if score > 0.8:
            ocr_texts.append((delete_non_alphanumeric(txt[0]), score))
        else:
            ocr_texts.append(("", 0.0))
    return ocr_texts


if __name__ == "__main__":
    # Example usage
    from fast_plate_ocr import LicensePlateRecognizer
    import numpy as np

    m = LicensePlateRecognizer('cct-s-v1-global-model')
    txt, score = m.run('Screenshot 2025-07-27 004223.png', return_confidence=True)
    print(txt, np.mean(score))
