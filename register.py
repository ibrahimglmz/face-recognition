import cv2
import mediapipe as mp
import time
import json
import os

# Kullanıcı adını al
kullanici = input("Kullanıcı Adı: ")

# MediaPipe yüz mesh modülünü başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmark noktaları için çizim özelliklerini tanımla
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Veri toplamaya başlama zamanı
start_time = time.time()

# Veri koleksiyonu
face_data_collection = []

# 3 saniye boyunca veri topla
while time.time() - start_time < 3:
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı.")
        continue

    # Görüntüyü RGB'ye çevir
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Yüz tespiti yap
    results = face_mesh.process(image_rgb)

    # Görüntüyü BGR'ye geri çevir
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Yüz landmark noktalarını çiz
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Her bir yüz için landmark verilerini topla
            face_data = {
                "timestamp": time.time(),
                "landmarks": []
            }
            for lm in face_landmarks.landmark:
                face_data["landmarks"].append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

                # Landmark noktalarını manuel olarak çiz
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)

            face_data_collection.append(face_data)

    # Görüntüyü göster
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Kamerayı ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()

# 'person' klasörünü oluştur (eğer yoksa)
os.makedirs('person', exist_ok=True)

# Verileri JSON formatında kaydet
dosya_adi = f'person/{kullanici}.json'
with open(dosya_adi, 'w') as f:
    json.dump(face_data_collection, f, indent=4)

print(f"Veriler '{dosya_adi}' dosyasına JSON formatında kaydedildi.")
