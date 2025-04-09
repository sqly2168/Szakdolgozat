import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
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
# Modellpéldány létrehozása és betöltése
embedding_dim = 128
num_classes = 4  # aron apa hanna és fanni

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resNet([3, 4, 6, 3], embedding_dim=embedding_dim).to(device)
classifier = nn.Linear(embedding_dim, num_classes).to(device)

model.load_state_dict(torch.load("trained_resnet_embeddings_four_person.pth"))
classifier.load_state_dict(torch.load("resnet_classifier_four_person.pth"))

model.eval()
classifier.eval()

#-------------------------------------------------------------
# Transzformáció
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # NumPy -> PIL konverzió! mivel PIL vár!
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(device)

def predict_class(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(img)
        output = classifier(embedding)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.squeeze().cpu().numpy()

#-------------------------------------------------------------
#futtatás mappában lévő képekkel
def batch_predict_folder(folder_path, class_names):
    """ Végigmegy a mappán és osztályoz minden képet """
    if not os.path.exists(folder_path):
        print(" nincs ilyen mappa")
        return

    print(f"\n {folder_path} mappa képei:\n")

    class_counts = [0 for i in class_names]

    for filename in os.listdir(folder_path):

        image_path = os.path.join(folder_path, filename)
        try:
            pred, probs = predict_class(image_path)
            class_counts[pred] += 1
            print(f"{filename} --> {class_names[pred]} ({probs[pred]*100:.2f}%)")
        except Exception as e:
            print(f"Hiba {filename} feldolgozása közben: {e}")
    
    for i, count in enumerate(class_counts):
        print(f"{class_names[i]}: {count} kép")

# Futtatás
if __name__ == "__main__":
    class_names = ["apa", "aron", "hanna", "fanni"]
    folder = "data/employees/fanni"

    batch_predict_folder(folder, class_names)
'''
#Futtatás anchorképpel
if __name__ == "__main__":
    class_names = ["apa", "aron", "hanna"]
    image_path = "anchor.jpg" 

    if not os.path.exists(image_path):
        print("nincs ilyen kep")
    else:
        pred, probs = predict_class(image_path)
        print(f"\n A kép osztálya: {class_names[pred]}")
        print(f"Valószínűségek: {probs}")
'''