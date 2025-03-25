from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT
import os
import cv2
from paddleocr import PaddleOCR
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

!wget https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf -P /usr/share/fonts/truetype/
!fc-cache -fv  

tracker = DeepSort(max_age=15, n_init=3)

# English-to-Arabic Mapping
english_to_arabic_mapping = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥',
    '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'هـ', 'F': 'ف', 'G': 'ق', 'H': 'ح',
    'I': 'ع', 'J': 'ج', 'K': 'ك', 'L': 'ل', 'M': 'م', 'N': 'ن', 'O': 'و', 'P': 'پ',
    'Q': 'ق', 'R': 'ر', 'S': 'س', 'T': 'ط', 'U': 'ع', 'V': 'ى', 'W': 'و', 'X': 'ص',
    'Y': 'ي', 'Z': 'ز'
}

def map_text(text, mapping_dict):
    return ''.join(mapping_dict.get(char, char) for char in text)

def is_valid_license_plate(text):
    pattern = r"^\d{4}[A-Z]{3}$"
    return re.match(pattern, text)

def detect_license_plate(image):
    MODEL_VERSION = "20230229"
    DEFAULT_FOLDER = os.path.join(os.path.expanduser("~"), ".hyperlpr3")
    det_model_path_640x = os.path.join(MODEL_VERSION, "onnx", "y5fu_640x_sim.onnx")
    det = MultiTaskDetectorORT(os.path.join(DEFAULT_FOLDER, det_model_path_640x), input_size=(640, 640))
    outputs = det(image)
    return [out[:4].astype(int) for out in outputs]

def extract_english_text(image, coordinates):
    ocr_english = PaddleOCR(use_angle_cls=True, lang='en')
    extracted_texts = []
    for idx, rect in enumerate(coordinates):
        cropped_image = image[rect[1]:rect[3], rect[0]:rect[2]]
        if cropped_image.size == 0:
            continue
        ocr_results_english = ocr_english.ocr(cropped_image, cls=True)
        if ocr_results_english and ocr_results_english[0]:
            result = ocr_results_english[0][0] if len(ocr_results_english[0]) < 2 else ocr_results_english[0][1]
            text = result[1][0].strip()
            confidence = result[1][1]
            if confidence > 0.80 and is_valid_license_plate(text):
                mapped_text = map_text(text, english_to_arabic_mapping)
                extracted_texts.append((f"{text} | {mapped_text}", rect))
    return extracted_texts

def add_text_to_frame(frame, text, box_coords, track_id=None):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/Amiri-Regular.ttf", 80)
    except IOError:
        font = ImageFont.load_default()

    x1, y1, x2, y2 = box_coords

    if track_id is not None:
        id_text = f"ID: {track_id}"
        id_position = (x1, y1 - 100) 
        draw.text(id_position, id_text, font=font, fill=(0,255,255))

    english_text, arabic_text = text.split(' | ')

    arabic_x = x1
    arabic_y = y1 - 80 
    for char in arabic_text:
        bbox = draw.textbbox((arabic_x, arabic_y), char, font=font)
        draw.text((arabic_x, arabic_y), char, fill=(255,0,0), font=font)
        arabic_x += (bbox[2] - bbox[0]) + 10

    english_y = y2 + 40  
    eng_bbox = draw.textbbox((x1, english_y), english_text, font=font)
    draw.text((x1, english_y), english_text, fill=(255,0,0), font=font)

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Main processing loop with tracking
video_path = "/content/2.mp4"
output_video_path = '/content/Deepoutput_video.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(output_video_path,
                         cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

    tracked_plates = {} 
    printed_plates = set()  

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        coordinates = detect_license_plate(frame)

        detections = []
        for rect in coordinates:
            x1, y1, x2, y2 = rect
            detections.append(([x1, y1, x2-x1, y2-y1], 0.9, 'license_plate'))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            if track.track_id not in tracked_plates:
                cropped = frame[y1:y2, x1:x2]
                text_data = extract_english_text(cropped, [(0, 0, cropped.shape[1], cropped.shape[0])])
                if text_data:
                    plate_text = text_data[0][0]
                    tracked_plates[track.track_id] = plate_text
                    
                    # Print only if not previously printed (with spaced Arabic text)
                    if plate_text not in printed_plates:
                        eng_part, ar_part = plate_text.split(' | ')
                        ar_spaced = ' '.join(ar_part)
                        print(f"New License Plate Detected: {eng_part} | {ar_spaced}")
                        printed_plates.add(plate_text)

            # Draw bounding box and text (original format)
            if track.track_id in tracked_plates:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                frame = add_text_to_frame(frame, tracked_plates[track.track_id], (x1, y1, x2, y2), track.track_id)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_video_path}")