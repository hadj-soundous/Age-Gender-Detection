import os

dataset_path = "dataset/UTKFace"

files = os.listdir(dataset_path)

print(f"Found {len(files)} images")

print("First 5 images:")

for f in files[:5]:
    print(f)