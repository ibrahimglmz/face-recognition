import cv2
import mediapipe as mp
import json
import numpy as np
import time

# Kullanıcı adını ve veri dosyasını belirt
kullanici = input("Kullanıcı Adı: ")
dosya_adi = f'person/{kullanici}.json'

# JSON dosyasını oku
try:
    with open(dosya_adi, 'r') as f:
        kayıtlı_veri = json.load(f)
except FileNotFoundError:
    print(f"Dosya bulunamadı: {dosya_adi}")
    exit(1)
except json.JSONDecodeError:
    print(f"JSON dosyası okunamadı: {dosya_adi}")
    exit(1)

# JSON dosyasının içeriğini kontrol et
if not isinstance(kayıtlı_veri, list) or len(kayıtlı_veri) == 0 or "landmarks" not in kayıtlı_veri[0]:
    print("JSON dosyası beklenen formatta değil.")
    exit(1)

# MediaPipe yüz mesh modülünü başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmark noktaları için çizim özelliklerini tanımla
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Veri karşılaştırma işlevi
def compare_landmarks(landmarks1, landmarks2, threshold=0.1):
    """
    Landmark verilerini karşılaştır.
    """
    if len(landmarks1) != len(landmarks2):
        return False

    for lm1, lm2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((lm1['x'] - lm2['x']) ** 2 + (lm1['y'] - lm2['y']) ** 2 + (lm1['z'] - lm2['z']) ** 2)
        if dist > threshold:
            return False

    return True

# Zamanlayıcılar ve bayraklar
start_time = time.time()
giriş_başarılı = False
uyarı_mesajı_yazıldı = False

while True:
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntüyü RGB'ye çevir
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Yüz tespiti yap
    results = face_mesh.process(image_rgb)

    # Görüntüyü BGR'ye geri çevir
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Her bir yüz için landmark verilerini topla
            face_data = {
                "landmarks": []
            }
            for lm in face_landmarks.landmark:
                face_data["landmarks"].append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

            # Karşılaştırma yap
            for veri in kayıtlı_veri:
                if compare_landmarks(face_data["landmarks"], veri["landmarks"]):
                    print("Giriş başarılı!")
                    giriş_başarılı = True
                    break

            if giriş_başarılı:
                break

            # Yüz landmark noktalarını çiz
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # 3 saniye içinde yüz algılanmadıysa veya uyuşmadıysa kamerayı kapat ve uyarı mesajı yazdır
    if not giriş_başarılı and not uyarı_mesajı_yazıldı and time.time() - start_time > 3:
        print("Uyarı: Yüz algılanamadı veya algılanan yüz kayıtlı veri ile uyuşmuyor.")
        uyarı_mesajı_yazıldı = True
        break

    # Görüntüyü göster
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

    # 'q' tuşuna basılınca çıkış yapma
    if cv2.waitKey(5) & 0xFF == 27 or giriş_başarılı:
        break

# Kamerayı ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()

if giriş_başarılı:
    print("Giriş başarılı!")
else:
    print("Giriş başarısız. Kamera kapatılıyor.")
