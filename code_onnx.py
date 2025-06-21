#!/usr/bin/env python3

import cv2
import numpy as np
import onnxruntime as rt
import time
import os
import math
import requests
# Pastikan lcd_display dan mqtt_client ada atau ganti dengan dummy jika tidak ada
try:
    from lcd_display import LCDDisplay
except ImportError:
    print("Warning: lcd_display not found. Skipping LCD display functionality.")
    class LCDDisplay:
        def __init__(self): pass
        def display_measurements(self, w, h): pass
        def cleanup(self): pass

try:
    from mqtt_client import MQTTClient
except ImportError:
    print("Warning: mqtt_client not found. Skipping MQTT functionality.")
    class MQTTClient:
        def __init__(self): pass
        def connect(self): pass
        def publish_measurement(self, height_cm, width_cm, confidence, class_id): pass
        def disconnect(self): pass


# Constants
MODEL_PATH = "model.onnx" # Ganti path ke file .onnx Anda
FIXED_DISTANCE = 200     # cm
CAMERA_FOV = 30.9        # derajat, sesuaikan dengan FOV horizontal kamera kamu
RESOLUTION_WIDTH = 224 # sesuaikan dengan resolusi input model (MODEL_INPUT_SIZE di Colab adalah 640x640, pastikan konsisten)

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

# Di Colab, MODEL_INPUT_SIZE adalah 640. Pastikan ini konsisten.
# Jika model ONNX Anda memang dilatih dengan input 224x224, maka RESOLUTION_WIDTH=224 sudah benar.
# Jika tidak, ubah RESOLUTION_WIDTH dan MODEL_INPUT_SIZE (di resize_with_padding) sesuai dengan model Anda.
PIXEL_TO_CM = hitung_pixel_to_cm(FIXED_DISTANCE, CAMERA_FOV, RESOLUTION_WIDTH)
MODEL_INPUT_SIZE = RESOLUTION_WIDTH # Pastikan ini sesuai dengan input_shape model ONNX Anda

def load_onnx_model(model_path):
    """Memuat model ONNX dan menginisialisasi sesi inferensi."""
    try:
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        print(f"Model ONNX berhasil dimuat. Nama input: '{input_name}'")
        
        # Tambahan: Cek dimensi input yang diharapkan model ONNX
        input_shape = session.get_inputs()[0].shape
        print(f"Model ONNX mengharapkan input shape: {input_shape}")
        if input_shape[1] != MODEL_INPUT_SIZE or input_shape[2] != MODEL_INPUT_SIZE:
            print(f"WARNING: MODEL_INPUT_SIZE ({MODEL_INPUT_SIZE}) tidak cocok dengan dimensi input model ONNX ({input_shape[1]}x{input_shape[2]}).")
            print("Pastikan MODEL_INPUT_SIZE di kode ini sesuai dengan model yang Anda konversi dari Colab.")
            print("Saat ini akan menggunakan MODEL_INPUT_SIZE yang didefinisikan di kode.")

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
    
    # 1. Resize dan padding gambar ke ukuran input model
    frame_resized = resize_with_padding(frame, MODEL_INPUT_SIZE)
    
    # 2. KONVERSI BGR ke RGB (PENTING: Model Anda dilatih dengan RGB)
    frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalisasi data (0-1) dan tambahkan dimensi batch
    input_data = frame_resized_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0) # Shape: (1, H, W, 3)
    
    # Menjalankan inferensi dengan ONNX Runtime
    result = session.run(None, {input_name: input_data})
    
    # Ambil output pertama dari hasil inferensi
    # Output Colab: predictions[0, ..., 0] -> (H, W) untuk heatmap probabilitas.
    # ONNX Runtime mungkin mengembalikan (1, H, W, 1) atau (1, H, W).
    raw_output_from_onnx = result[0] 
    
    # Debugging: print shape output mentah
    print(f"Shape output mentah dari ONNX: {raw_output_from_onnx.shape}")

    # Ekstrak mask 2D dari output
    # Asumsi output model adalah (batch_size, H, W, 1) atau (batch_size, H, W)
    if len(raw_output_from_onnx.shape) == 4 and raw_output_from_onnx.shape[3] == 1:
        # Jika bentuknya (1, H, W, 1), ambil H, W
        raw_mask_2d = raw_output_from_onnx[0, :, :, 0]
    elif len(raw_output_from_onnx.shape) == 3 and raw_output_from_onnx.shape[0] == 1:
        # Jika bentuknya (1, H, W), ambil H, W
        raw_mask_2d = raw_output_from_onnx[0, :, :]
    else: # Asumsi sudah (H, W) jika ONNX runtime meng-squeeze dimensinya
        raw_mask_2d = raw_output_from_onnx
    
    # Debugging: print min/max/mean dari raw_mask_2d
    print(f"Min/Max/Mean raw_mask_2d: {np.min(raw_mask_2d):.4f} / {np.max(raw_mask_2d):.4f} / {np.mean(raw_mask_2d):.4f}")

    # 4. KOREKSI LOGIKA UTAMA: Balikkan peta probabilitas karena model Anda mendeteksi background.
    # Ini harus menghasilkan probabilitas objek, di mana nilai tinggi = objek.
    probability_map = 1 - raw_mask_2d
    
    # 5. Lakukan thresholding untuk mendapatkan mask biner (objek = 255, background = 0)
    # Ambang batas 0.5 lebih umum untuk segmentasi, tetapi Anda bisa coba 0.7 seperti di Colab Anda.
    threshold = 0.5 # Atau gunakan 0.7 seperti di Colab Anda
    mask = (probability_map > threshold).astype(np.uint8) * 255
    
    # Debugging: print persentase piksel di atas threshold
    detected_pixels_after_threshold = np.sum(mask > 0)
    total_pixels_in_mask = mask.size
    print(f"Persentase piksel terdeteksi setelah threshold ({threshold}): {(detected_pixels_after_threshold / total_pixels_in_mask) * 100:.2f}%")

    # Resize mask kembali ke ukuran frame asli
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # PENTING: Coba HAPUS `cv2.bitwise_not(mask)` ini.
    # Jika mask Anda sekarang sudah 255 untuk objek dan 0 untuk background,
    # maka `bitwise_not` akan membalikkannya, yang mungkin tidak Anda inginkan.
    # Berdasarkan visualisasi Colab, mask objek berwarna putih (yang berarti nilai tinggi).
    # Jika Anda ingin kontur menemukan area putih, maka biarkan mask apa adanya.
    # Jika Anda tetap ingin membalikkannya (misal, untuk latar belakang hitam), pastikan itu disengaja.
    # Saya akan mengomentarinya untuk percobaan pertama.
    # mask = cv2.bitwise_not(mask) # <-- Coba nonaktifkan baris ini dulu!
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_frame = frame.copy()
    
    confidence_percentage = 0.0 # Default value jika tidak ada deteksi
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Hitung rata-rata confidence score dalam area objek yang terdeteksi
        # Pastikan probability_map sudah di-resize ke ukuran frame asli juga
        probability_map_resized = cv2.resize(probability_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Buat mask biner di ukuran asli untuk mengambil nilai confidence
        binary_mask_resized = (probability_map_resized > threshold).astype(np.uint8)
        
        # Pastikan area yang dihitung confidence_percentage adalah area objek yang benar
        if np.sum(binary_mask_resized) > 0:
            confidence_percentage = np.mean(probability_map_resized[binary_mask_resized > 0]) * 100
        else:
            confidence_percentage = 0.0

        # Visualisasi mask hijau di atas objek
        mask_color = np.zeros_like(frame)
        # Warna hijau untuk area objek yang terdeteksi (di mana mask > 0)
        mask_color[mask > 0] = [0, 255, 0] # BGR format
        result_frame = cv2.addWeighted(result_frame, 0.7, mask_color, 0.3, 0)
        
        # Gambar bounding box (merah)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Hitung pengukuran dalam cm
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        # Kirim ke MQTT
        mqtt_client.publish_measurement(
            height_cm=height_cm,
            width_cm=width_cm,
            confidence=confidence_percentage / 100.0, # Kirim confidence dalam desimal (0-1)
            class_id=1
        )
        
        # Tampilkan teks pengukuran dan confidence
        measurements = f"W: {width_cm:.1f}cm H: {height_cm:.1f}cm"
        confidence_text = f"Conf: {confidence_percentage:.1f}%"
        
        # Posisi teks: pengukuran di atas bbox, confidence di bawah bbox atau di samping
        text_size_meas = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (x, y - text_size_meas[1] - 10), (x + text_size_meas[0], y), (0, 0, 0), -1)
        cv2.putText(result_frame, measurements, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Kuning
        
        text_size_conf = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (x, y + h + 5), (x + text_size_conf[0], y + h + text_size_conf[1] + 15), (0, 0, 0), -1)
        cv2.putText(result_frame, confidence_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Biru Muda
        
        print(f"\nPengukuran tubuh pada jarak {FIXED_DISTANCE}cm:")
        print(f"Lebar: {width_cm:.1f} cm")
        print(f"Tinggi: {height_cm:.1f} cm")
        print(f"Dimensi piksel: {w}x{h}")
        print(f"Persentase Keyakinan Objek: {confidence_percentage:.2f}%")
        
        if lcd_display is not None:
            lcd_display.display_measurements(width_cm, height_cm)
            # Anda mungkin ingin juga menampilkan confidence di LCD jika ada baris tambahan
            # atau menggabungkan dengan tinggi/lebar.
    else:
        print("\nTidak ada objek terdeteksi.")
        if lcd_display is not None:
            lcd_display.display_measurements(0, 0) # Atau tampilkan pesan "Tidak terdeteksi"

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
    # Coba berbagai backend kamera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # V4L2 sering kali lebih baik di RPi
    if not cap.isOpened():
        cap = cv2.VideoCapture(0) # Fallback ke default
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka. Pastikan kamera terhubung dan diaktifkan.")
        return

    # Set resolusi kamera (pastikan ini didukung oleh kamera Anda)
    # Ini adalah resolusi frame yang diambil, bukan input model.
    # Usahakan set resolusi yang mendekati rasio aspek model atau kamera Anda.
    # Misal: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Tekan SPASI untuk mengambil gambar, atau 'q' untuk keluar.")
    frame = None
    while True:
        ret, preview = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break
        # Rotasi gambar jika kamera terpasang vertikal (seperti di gambar Anda)
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