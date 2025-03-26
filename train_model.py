import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet import resNet  # saját ResNet modell importálása

# Eszköz kiválasztása
device = "cuda" if torch.cuda.is_available() else "cpu"

# Adatok transzformációja
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Datasetek betöltése
train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
val_dataset = datasets.ImageFolder(root="data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f" Train minták: {len(train_dataset)} | Val minták: {len(val_dataset)}")

# Modell példányosítása
model = resNet([3, 4, 6, 3], embedding_dim=128).to(device)

# Osztályozáshoz teljesen összekapcsolt réteg
classifier = nn.Linear(128, len(train_dataset.classes)).to(device)

# Loss és optimizáló
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.0001)

# Tanítási ciklus
def train(model, classifier, loader):
    model.train()
    classifier.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        embeddings = model(images)
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Validációs függvény
def evaluate(model, classifier, loader):
    model.eval()
    classifier.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            outputs = classifier(embeddings)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# Epoch-ok futtatása
num_epochs = 30
for epoch in range(num_epochs):
    loss = train(model, classifier, train_loader)
    acc = evaluate(model, classifier, val_loader)
    print(f"\n Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

# Modell mentése
torch.save(model.state_dict(), "trained_resnet_embeddings_sajatkepek.pth")
torch.save(classifier.state_dict(), "resnet_classifier_sajatkepek.pth")
print("\n Modell és osztályozó elmentve!")
