#!/usr/bin/env python3

import cv2
import numpy as np
import onnxruntime as rt # Ganti TensorFlow dengan ONNX Runtime
import time
import os
import math
import requests
from lcd_display import LCDDisplay
from mqtt_client import MQTTClient

# Constants
MODEL_PATH = "mask_rcnn_model.onnx" # Ganti path ke file .onnx Anda
FIXED_DISTANCE = 200  # cm
CAMERA_FOV = 30.9     # derajat, sesuaikan dengan FOV horizontal kamera kamu
RESOLUTION_WIDTH = 480 # sesuaikan dengan resolusi input model

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Initialize MQTT client
mqtt_client = MQTTClient()
mqtt_client.connect()

def hitung_pixel_to_cm(jarak_cm, fov_derajat, resolusi_horizontal):
    """Hitung nilai cm per pixel dari jarak dan FOV"""
    fov_rad = math.radians(fov_derajat / 2)
    lebar_cm = 2 * math.tan(fov_rad) * jarak_cm
    return lebar_cm / resolusi_horizontal

PIXEL_TO_CM = hitung_pixel_to_cm(FIXED_DISTANCE, CAMERA_FOV, RESOLUTION_WIDTH)
MODEL_INPUT_SIZE = 640

def load_onnx_model(model_path):
    """Memuat model ONNX dan menginisialisasi sesi inferensi."""
    try:
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        print(f"Model ONNX berhasil dimuat. Nama input: '{input_name}'")
        return session, input_name
    except Exception as e:
        print(f"Error memuat model ONNX: {e}")
        return None, None

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def process_frame(frame, session, input_name, lcd_display=None):
    """Memproses frame tunggal untuk segmentasi dan pengukuran tubuh menggunakan ONNX."""
    
    frame_resized = resize_with_padding(frame, MODEL_INPUT_SIZE)
    
    input_data = frame_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Menjalankan inferensi dengan ONNX Runtime
    result = session.run(None, {input_name: input_data})
    mask = result[0][0] # Ambil output pertama dari hasil inferensi

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    mask = (mask > 0.7).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_frame = frame.copy()
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        mask_color = np.zeros_like(frame)
        mask_color[mask > 0] = [0, 255, 0]
        result_frame = cv2.addWeighted(result_frame, 0.7, mask_color, 0.3, 0)
        
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        mqtt_client.publish_measurement(
            height_cm=height_cm,
            width_cm=width_cm,
            confidence=1.0, # Anda bisa mengambil skor keyakinan jika model ONNX menyediakannya
            class_id=1
        )
        
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        text_size = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 0), -1)
        cv2.putText(result_frame, measurements, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        print(f"\nPengukuran tubuh pada jarak {FIXED_DISTANCE}cm:")
        print(f"Lebar: {width_cm:.1f} cm")
        print(f"Tinggi: {height_cm:.1f} cm")
        print(f"Dimensi piksel: {w}x{h}")
        
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
    
    return result_frame

def main():
    # Pastikan model ONNX ada
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File model tidak ditemukan di '{MODEL_PATH}'")
        return

    # Muat model ONNX
    session, input_name = load_onnx_model(MODEL_PATH)
    if session is None:
        return

    # Inisialisasi LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display berhasil diinisialisasi")
    except Exception as e:
        print(f"Error menginisialisasi LCD display: {e}")
        lcd = None

    # Inisialisasi kamera
    print("Menghidupkan kamera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka.")
        return

    print("Tekan SPASI untuk mengambil gambar, atau 'q' untuk keluar.")
    frame = None
    while True:
        ret, preview = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        preview = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Preview Kamera", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # tombol SPASI ditekan
            frame = preview.copy()
            print("Gambar diambil.")
            break
        elif key == ord('q'):
            print("Keluar tanpa mengambil gambar.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow("Preview Kamera")

    if frame is not None:
        print("Memproses gambar...")
        result = process_frame(frame, session, input_name, lcd)
        cv2.imshow("Hasil Deteksi", result)
        print("Tekan tombol apa saja untuk keluar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if lcd is not None:
        lcd.cleanup()
    mqtt_client.disconnect()

if __name__ == "__main__":
    main()