import cv2                              #cv2 kamera
import os                               # Mappak konyvtarak etc.       
import random                           #random szamok
import numpy as np                      #matek
from matplotlib import pyplot as plt    #fgv kirajzolasa
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" #ez csak egy warning-ra van, nem kell figyelni

# Tensorflow dependencyk - Functional API
from tensorflow.python.keras.models import Model        
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU') # GPU-k listazasa
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True) #gpu memoriajanak hasznalata
    
for gpu in gpus: 
    print(gpu) # GPU-k listazasa

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))