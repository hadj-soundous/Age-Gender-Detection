import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/age_gender_model.keras")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    gender_pred, age_pred = model.predict(img, verbose=0)

    gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
    age = int(age_pred[0][0])

    text = f"{gender}, {age}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()