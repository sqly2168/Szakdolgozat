import os
import shutil

# Forrás mappa
src_dir = "data/fanni"

# Cél mappák
train_dir = "data/train/fanni"
val_dir = "data/val/fanni"

# Célmappák létrehozása, ha nem léteznek
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Fájlok beolvasása és rendezése név szerint
all_images = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Képek szétválogatása
for i, img in enumerate(all_images):
    src_path = os.path.join(src_dir, img)

    if (i + 1) % 4 == 0:
        dst_path = os.path.join(val_dir, img)
    else:
        dst_path = os.path.join(train_dir, img)

    shutil.copy2(src_path, dst_path)  # vagy .move ha áthelyezni akarod

print(f"{len(all_images)} kép feldolgozva. Kész!")
