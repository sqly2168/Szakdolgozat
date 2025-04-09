import cv2                              #cv2 kamera
import os                               # Mappak konyvtarak etc.       
import random                           #random szamok
import numpy as np                      #matek
from matplotlib import pyplot as plt    #fgv kirajzolasa
import torch

# Mappa az új személy képeihez
save_dir = "data/fanni"
os.makedirs(save_dir, exist_ok=True)

# Szokásos módon kell a referenciakép, meg lehet adni lokálisan, ha nincs megadva akkor kamerából olvasok
anchor_img_path = "semmi.jpg"
#anchor_img_path = "C:\\Users\\SQLY\\Desktop\\ow.png"
anchor_img = cv2.imread(anchor_img_path) 

# Ha van meglévő anchor kép, akkor megjelenítjük
if anchor_img is not None:
    print("Referenciakép betöltve.")
    cv2.imshow("Referenciakép (Anchor): ", anchor_img)
    cv2.waitKey(3000)  # 3 másodpercig mutatja az anchort
else:
    print("Kamerás Anchor kép készítése...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hiba: Nem sikerült megnyitni a kamerát!")  # töltelékkód a szakdogába
    exit()

image_counter = 0  # számláló az egyedi fájlnevekhez

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hiba: Nem sikerült beolvasni a kameraképet!")  # töltelékkód a szakdogába2
        break

    # Kép mérete
    h, w, _ = frame.shape  # Magasság, szélesség, csatornák (RGB)

    # Középpont koordinátái
    center_x, center_y = w // 2, h // 2  

    # Kivágási koordináták
    half_size = 125  # 250 / 2
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size

    # Középső 250x250-es rész kivágása (ha lehetséges)
    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        cropped_frame = frame[y1:y2, x1:x2]
    else:
        cropped_frame = frame

    cv2.imshow("Cheese! ---> nyomd meg a 0-ás gombot és elkészül az anchor", frame)

    pressedkey = cv2.waitKey(1) & 0xFF

    # '0' → anchor kép mentése
    if pressedkey == ord('0'):
        anchor_img = cropped_frame.copy()
        cv2.imwrite("anchor.jpg", anchor_img)
        print("Anchor kép frissítve!")
        cv2.imshow("Referenciakép (Anchor): ", anchor_img)

    # 't' → új személy kép mentése egyedi névvel
    if pressedkey == ord('t'):
        image_filename = f"fanni_{image_counter:04d}.jpg"
        full_path = os.path.join(save_dir, image_filename)
        cv2.imwrite(full_path, cropped_frame)
        print(f"Új személy képe mentve: {full_path}")
        image_counter += 1

    # 'q' → kilépés
    if pressedkey == ord('q'):
        break

# Kamera bezárása, RAM felszabadítása
cap.release()
cv2.destroyAllWindows()
