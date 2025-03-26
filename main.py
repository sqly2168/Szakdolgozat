import cv2                              #cv2 kamera
import os                               # Mappak konyvtarak etc.       
import random                           #random szamok
import numpy as np                      #matek
from matplotlib import pyplot as plt    #fgv kirajzolasa
import torch

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

    # Képkivágás (ellenőrizzük, hogy a kamera elég nagy-e)
    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        cropped_frame = frame[y1:y2, x1:x2]  # Középső 250x250-es rész kivágása
    else:
        cropped_frame = frame  # Ha valamiért nem lehet kivágni, az eredeti képet használjuk

    cv2.imshow("Cheese! ---> nyomd meg a 0-ás gombot és elkészül az anchor", frame)

    pressedkey = cv2.waitKey(1) & 0xFF
    
    # Ha '0'-t nyomunk, akkor az aktuális kameraképből kivágott részt menti az anchor_img változóba
    if pressedkey == ord('0'):
        anchor_img = cropped_frame.copy()  # Középső rész mentése
        cv2.imwrite("anchor.jpg", anchor_img)  # Kép mentése a mappába
        print("Anchor kép frissítve!")
        cv2.imshow("Referenciakép (Anchor): ", anchor_img)
    
    # Kilépés a 'q' billentyűvel
    if pressedkey == ord('q'):
        break

# Kamera bezárása, RAM felszabadítása -- ajánlott
cap.release()
cv2.destroyAllWindows()
