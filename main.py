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

if anchor_img is None:
    print("Kamerás Anchor kép készítése...")
else:
    print("Referenciakép betöltve.")
    cv2.imshow("Referenciakép (Anchor): ", anchor_img)
    cv2.waitKey(3000)  # 3 másodpercig mutatja az anchort


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hiba: Nem sikerült megnyitni a kamerát!") #töltelékkód a szakdogába
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hiba: Nem sikerült beolvasni a kameraképet!") #töltelékkód a szakdogába2
        break
    
    cv2.imshow("Cheese! ---> nyomd meg a 0-ás gombot és elkészül az anchor", frame)

    pressedkey = cv2.waitKey(1) & 0xFF
    # Ha '0'-t nyomunk, akkor az aktuális kameraképet menti az anchor_img változóba
    if pressedkey == ord('0'):
        anchor_img = frame.copy()              #így nem kell bezárni a kamerát hogy menthessem a képen
        cv2.imwrite("anchor.jpg", anchor_img)  # Kép mentése a mappába is
        print("Anchor kép frissítve!")
        cv2.imshow("Referenciakép (Anchor): ", anchor_img)
    
    # Kilépés a q-val
    if pressedkey == ord('q'):
        break

# Kamera bezárása, ram felszabadítása --ajánlott
cap.release()
cv2.destroyAllWindows()



#először szinesek
#frekvencia diagramm
