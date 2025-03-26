import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import os
import pickle
from numpy.linalg import norm

#-------------------------------------------------------------
# Konvolúciós blokk
class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c(x))

#-------------------------------------------------------------
# Residual blokk
class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        res_channels = in_channels // 4
        stride = 1

        self.projection = in_channels != out_channels
        if self.projection:
            self.p = convBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2

        if first:
            self.p = convBlock(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channels = in_channels

        self.c1 = convBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = convBlock(res_channels, res_channels, 3, stride, 1)
        self.c3 = convBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h

#-------------------------------------------------------------
# ResNet modell
class resNet(nn.Module):
    def __init__(self, no_blocks, in_channels=3, embedding_dim=128):
        super().__init__()
        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([residualBlock(64, 256, True)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(residualBlock(out_features[i-1], out_features[i]))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(residualBlock(out_features[i], out_features[i]))

        self.conv1 = convBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embedding_dim)  # Embedding mérete
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Visszatérés embeddingként
        return x

#-------------------------------------------------------------
# Modellpéldány létrehozása
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resNet([3, 4, 6, 3]).to(device)

#-------------------------------------------------------------
# Képfeldolgozási transzformációk
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalizálás (-1 és 1 között)
])

def preprocess_image(image_path):
    """Betölti, átméretezi és normalizálja a képet"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0)  # Extra dimenzió hozzáadása
    return img.to(device)

def get_embedding(model, image_path):
    """ Kiszámolja az embedding vektort egy képre """
    img = preprocess_image(image_path)

    model.eval()
    with torch.no_grad():
        embedding = model(img)

    return embedding.cpu().squeeze(0).numpy()  # 1D vektorrá alakítás

#-------------------------------------------------------------
# Koszinusz hasonlóság számítás
def cosine_similarity(vec1, vec2, epsilon=1e-8):
    """ Koszinusz hasonlóság két embedding vektor között """
    vec1, vec2 = vec1.flatten(), vec2.flatten()
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + epsilon)

#-------------------------------------------------------------
# Anchor kép betöltése és embedding generálás
anchor_path = "anchor.jpg"
if os.path.exists(anchor_path):
    anchor_embedding = get_embedding(model, anchor_path)
    print(" Anchor embedding létrehozva!")
else:
    print(" Hiba: Anchor kép nem található!")
    exit()

#-------------------------------------------------------------
# Pozitív és negatív képek összehasonlítása
positive_folder = "data/positive"
negative_folder = "data/negative"
threshold = 0.6  # Ha 60% felett van, akkor egyező arcok

print("\n[P] Pozitív képek ellenőrzése:")
for img_name in os.listdir(positive_folder):
    img_path = os.path.join(positive_folder, img_name)
    test_embedding = get_embedding(model, img_path)
    similarity = cosine_similarity(anchor_embedding, test_embedding)
    result = " EGYEZIK" if similarity > threshold else " NEM EGYEZIK"
    print(f" {img_name} - Hasonlóság: {similarity:.4f} - {result}")

print("\n[N] Negatív képek ellenőrzése:")
for img_name in os.listdir(negative_folder):
    img_path = os.path.join(negative_folder, img_name)
    test_embedding = get_embedding(model, img_path)
    similarity = cosine_similarity(anchor_embedding, test_embedding)
    result = " NEM EGYEZIK" if similarity < threshold else " HIBA"
    print(f" {img_name} - Hasonlóság: {similarity:.4f} - {result}")
