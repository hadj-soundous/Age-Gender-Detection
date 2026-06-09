import os
import cv2
import numpy as np

DATASET_PATH = r"C:\Users\DELL\Downloads\archive\UTKFace"

images = []
ages = []
genders = []

for filename in os.listdir(DATASET_PATH):

    try:
        parts = filename.split("_")

        age = int(parts[0])
        gender = int(parts[1])

        img_path = os.path.join(DATASET_PATH, filename)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (224, 224))

        image = image / 255.0

        images.append(image)
        ages.append(age)
        genders.append(gender)

    except (ValueError, IndexError, cv2.error, OSError) as e:
        # Skip files that don't match expected naming or can't be read/processed
        continue

X = np.array(images, dtype=np.float32)
ages = np.array(ages)
genders = np.array(genders)

print("Images:", X.shape)
print("Ages:", ages.shape)
print("Genders:", genders.shape)

np.save("X.npy", X)
np.save("ages.npy", ages)
np.save("genders.npy", genders)

print("Dataset saved successfully.")