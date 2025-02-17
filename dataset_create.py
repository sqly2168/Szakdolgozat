import os                               # Mappak konyvtarak etc.       

POS_path = os.path.join("data", "positive") #képek data/positive
NEG_path = os.path.join("data", "negative")
ANC_path = os.path.join("data", "anchor")

os.makedirs(POS_path)
os.makedirs(NEG_path)
os.makedirs(ANC_path)


