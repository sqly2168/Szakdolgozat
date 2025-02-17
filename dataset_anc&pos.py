import uuid
import cv2
import os

POS_path = os.path.join("data", "positive") #képek data/positive
NEG_path = os.path.join("data", "negative")
ANC_path = os.path.join("data", "anchor")

# kamera megnyitása
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # frame levágás 250x250 és a közepére helyezzük a keretet
    frame = frame[120:120+250,200:200+250, :]
    
    # anchor kép
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # egyedi fájlnév létrehozása
        imgname = os.path.join(ANC_path, '{}.jpg'.format(uuid.uuid1()))
        # kép mentése
        cv2.imwrite(imgname, frame)
    
    # positive kép
    if cv2.waitKey(1) & 0XFF == ord('p'): 
        imgname = os.path.join(POS_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # kamerakép megjelenítése
    cv2.imshow('Image Collection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()