import os
import io
import time
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from util import read_license_plate


def load_models():
    coco_model = YOLO('yolov8n.pt')
    lp_model_path = os.path.join('models', 'license_plate_detector.pt')
    license_plate_model = None
    if os.path.exists(lp_model_path):
        try:
            license_plate_model = YOLO(lp_model_path)
        except Exception as e:
            st.warning(f"Failed to load license plate model: {e}")
            license_plate_model = None
    else:
        st.info("License plate model not found at 'models/license_plate_detector.pt'. Vehicle-only detection will run.")
    return coco_model, license_plate_model


def detect_vehicles(frame: np.ndarray, coco_model: YOLO, vehicle_class_ids: List[int]) -> List[Tuple[float, float, float, float, float]]:
    results = coco_model(frame)[0]
    vehicles = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicle_class_ids:
            vehicles.append((x1, y1, x2, y2, score))
    return vehicles


def detect_license_plates(frame: np.ndarray, license_plate_model: YOLO) -> List[Tuple[float, float, float, float, float]]:
    if license_plate_model is None:
        return []
    results = license_plate_model(frame)[0]
    plates = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        plates.append((x1, y1, x2, y2, score))
    return plates


def draw_detections(frame: np.ndarray,
                    vehicles: List[Tuple[float, float, float, float, float]],
                    plates: List[Tuple[float, float, float, float, float]],
                    show_text: bool = True) -> np.ndarray:
    annotated = frame.copy()
    # Draw vehicles
    for (x1, y1, x2, y2, score) in vehicles:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated, f"vehicle {score:.2f}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Draw license plates + OCR
    for (x1, y1, x2, y2, score) in plates:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # OCR
        lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        if lp_crop.size == 0:
            continue
        gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        text, text_score = read_license_plate(thresh)
        if show_text and text is not None:
            label = f"{text} ({text_score:.2f})"
            cv2.putText(annotated, label, (int(x1), max(int(y1) - 28, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return annotated


def process_image(image_bgr: np.ndarray, coco_model: YOLO, license_plate_model: YOLO, vehicle_class_ids: List[int]) -> np.ndarray:
    vehicles = detect_vehicles(image_bgr, coco_model, vehicle_class_ids)
    plates = detect_license_plates(image_bgr, license_plate_model)
    annotated = draw_detections(image_bgr, vehicles, plates)
    return annotated


def process_video(file_bytes: bytes, coco_model: YOLO, license_plate_model: YOLO, vehicle_class_ids: List[int]):
    # Write to a temporary file for OpenCV
    temp_path = 'uploaded_video.mp4'
    with open(temp_path, 'wb') as f:
        f.write(file_bytes)

    cap = cv2.VideoCapture(temp_path)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_image(frame, coco_model, license_plate_model, vehicle_class_ids)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB')
        current += 1
        progress_bar.progress(min(current / total_frames, 1.0))
    cap.release()
    os.remove(temp_path)


def main():
    st.set_page_config(page_title='ANPR - YOLOv8', layout='wide')
    st.title('Automatic Number Plate Recognition (YOLOv8)')
    st.caption('Basic web interface for vehicle detection and license plate reading')

    coco_model, license_plate_model = load_models()
    vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)

    mode = st.sidebar.radio('Mode', ['Image', 'Video'])

    if mode == 'Image':
        uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error('Failed to read image')
                return
            annotated = process_image(img_bgr, coco_model, license_plate_model, vehicles)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels='RGB')

    else:
        uploaded = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'mkv'])
        if uploaded is not None:
            process_video(uploaded.read(), coco_model, license_plate_model, vehicles)


if __name__ == '__main__':
    main()


