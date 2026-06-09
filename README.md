# Age & Gender Detection AI

A Deep Learning project that predicts a person's **Age** and **Gender** from facial images using **TensorFlow**, **MobileNetV2**, and **OpenCV**.

## Overview

This project trains a multi-output neural network on the UTKFace dataset to:

* Predict Gender (Male / Female)
* Estimate Age
* Run predictions on images
* Perform real-time webcam detection

The model uses **Transfer Learning** with MobileNetV2 to improve performance and reduce training time.

---

## Project Structure

```text
AgeGenderAI/
│
├── dataset/
│   └── UTKFace/
│
├── models/
│   └── age_gender_model.keras
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── webcam.py
│   └── test_dataset.py
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── X.npy
├── ages.npy
└── genders.npy
```

---

## Technologies Used

* Python 3.11
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Scikit-Learn

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/hadj-soundous/Age-Gender-Detection.git
cd AgeGenderAI
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the UTKFace dataset.

After downloading the dataset, place it inside:

```text
dataset/UTKFace/
```

Example:

```text
dataset/UTKFace/
├── 1_0_0_20170109150557335.jpg
├── 25_1_0_20170116174525125.jpg
├── ...
```

Filename format:

```text
age_gender_race_timestamp.jpg
```

Example:

```text
25_1_0_20170116174525125.jpg
```

Where:

* Age = 25
* Gender = Female (1)

---

## Verify Dataset

Run:

```bash
python src/test_dataset.py
```

Expected output:

```text
Found 23708 images
```

---

## Data Preprocessing

Convert images into NumPy arrays:

```bash
python src/preprocess.py
```

Generated files:

```text
X.npy
ages.npy
genders.npy
```

---

## Model Training

Train the model:

```bash
python src/train.py
```

The trained model will be saved as:

```text
models/age_gender_model.keras
```

---

## Image Prediction

Place a test image in the project root.

Example:

```text
test.jpg
```

Run:

```bash
python src/predict.py
```

Example output:

```text
Predicted Gender: Female
Predicted Age: 24
```

---

## Real-Time Webcam Detection

Run:

```bash
python src/webcam.py
```

The application will:

1. Open your webcam
2. Capture live video
3. Predict age and gender
4. Display predictions on the screen

Press:

```text
Q
```

to quit.

---

## Model Architecture

```text
Input Image
     ↓
MobileNetV2
     ↓
GlobalAveragePooling
     ↓
Dense Layers
     ↓
 ┌──────────────┴──────────────┐
 ↓                             ↓
Gender Output             Age Output
(Binary)                 (Regression)
```

---

## Results

Typical performance on UTKFace:

* Gender Accuracy: ~90–97%
* Age MAE: ~3–6 years

Performance may vary depending on training settings and hardware.

---

## Future Improvements

* Face Detection before prediction
* Better age estimation
* Streamlit Web App
* Model Fine-Tuning
* Improved webcam performance
* Deployment to cloud platforms

---

## Author

**Soundous Hadj**

Computer Science Student – Artificial Intelligence & Applications

ENSTA Algeria
