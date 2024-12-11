from flask import Blueprint, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import base64
from datetime import datetime
import re



ir_ocr = Blueprint('ir_ocr', __name__, url_prefix='/')

# Load YOLO model for IR counting and object detection
model = YOLO('ir_ocr/model/finalYolo220.pt')

# Tesseract path (update as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Date extraction and cleaning functions
date_formats = {
    "%d%b%Y": r"\b\d{2}\s?[A-Za-z]{3}\s?\d{4}\b",
    "%b%d%Y": r"\b[A-Za-z]{3}\s?\d{2}\s?\d{4}\b",
    "%b%Y": r"\b[A-Za-z]{3}\s?\d{4}\b(?!\d)",
    "%d%b%y": r"\b\d{2}\s?[A-Za-z]{3}\s?\d{2}\b",
    "%d%m%Y": r"\b\d{2}\s?\d{2}\s?\d{4}\b",
    "%m%Y": r"\b\d{2}\s?\d{4}\b(?!\d)",
    "%Y%m%d": r"\b\d{4}\s?\d{2}\s?\d{2}\b",
    "%Y%m": r"\b\d{4}\s?\d{2}\b(?!\d)",
    "%m%d": r"\b\d{2}\s?\d{2}\b(?!\d)"
}

def extract_dates(text):
    detected_dates = set()
    for date_format, pattern in date_formats.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                parsed_date = datetime.strptime(match.replace(" ", ""), date_format)
                detected_dates.add(parsed_date)
            except ValueError:
                continue
    return sorted(detected_dates)

def clean_text(text):
    replacements = {'O': '0', 'I': '1', 'Z': '2', 'l': '1', 'S': '5', 's': '5'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def determine_expiry_and_shelf_life(detected_dates):
    if not detected_dates:
        return "No expiry date found","Can Not Be calculated";
    expiry_date = max(detected_dates)
    today = datetime.today()
    if expiry_date < today:
        return expiry_date.strftime('%d %b %Y'), "Expired"
    else:
        remaining_days = (expiry_date - today).days
        return expiry_date.strftime('%d %b %Y'), f"{remaining_days} days remaining"

@ir_ocr.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = model.predict(source=img)
            detected_objects = {}
            ocr_results = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

                    roi = img[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    processed_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    ocr_text = pytesseract.image_to_string(processed_roi, config='--psm 6').strip()
                    text = clean_text(ocr_text)
                    detected_dates = extract_dates(text)
                    expiry_date, shelf_life_status = determine_expiry_and_shelf_life(detected_dates)

                    ocr_results.append({
                        "class": class_name,
                        
                        "expiry_date": expiry_date,
                        "shelf_life": shelf_life_status
                    })

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return render_template('index.html', detected_objects=detected_objects, ocr_results=ocr_results, image=img_base64)

    return render_template('index.html', detected_objects=None, ocr_results=None, image=None)
