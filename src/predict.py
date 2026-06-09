import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("models/age_gender_model.keras")

IMAGE_PATH = "test.jpg"

image = cv2.imread(IMAGE_PATH)

image = cv2.resize(image, (224,224))

image = image / 255.0

image = np.expand_dims(image, axis=0)

gender_pred, age_pred = model.predict(image)

gender = "Female" if gender_pred[0][0] > 0.5 else "Male"

age = int(age_pred[0][0])

print("Predicted Gender:", gender)
print("Predicted Age:", age)