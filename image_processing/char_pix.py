import cv2
import pytesseract
from pytesseract import Output
from char_cnn import CNNModel
import torch
import torchvision.transforms as T
import math
def extract_char_boxes(image_path, psm=6, oem=3, preprocess=True):
    """
    Returns:
        char_crops: list of dicts:
            - char: recognized character (string)
            - bbox: (x, y, w, h) in top-left origin
            - pixels: NumPy array (H, W, 3) BGR crop
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not load image: {image_path}")
    H, W = img_bgr.shape[:2]
    if preprocess:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb_for_ocr = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    else:
        rgb_for_ocr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    config = f'--psm {psm} --oem {oem}'
    boxes = pytesseract.image_to_boxes(rgb_for_ocr, output_type=Output.DICT, config=config)
    char_crops = []
    centers=[]
    n = len(boxes.get("char", []))
    for i in range(n):
        ch   = boxes["char"][i]
        left = int(boxes["left"][i])
        bottom = int(boxes["bottom"][i])
        right = int(boxes["right"][i])
        top = int(boxes["top"][i])
        x1, y1 = left,  H - top
        x2, y2 = right, H - bottom
        x1c, y1c = max(0, min(W, x1)), max(0, min(H, y1))
        x2c, y2c = max(0, min(W, x2)), max(0, min(H, y2))
        if x2c > x1c and y2c > y1c:
            crop = img_bgr[y1c:y2c, x1c:x2c].copy()
            x_c=x1c+(x2c-x1c)//2
            y_c=y1c+(y2c-y1c)//2
            centers.append([float(x_c)/W,float(y_c)/H])
            char_crops.append(crop)
    return centers,char_crops