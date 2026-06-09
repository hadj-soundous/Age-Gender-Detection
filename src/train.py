import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Loading data...")

X = np.load("X.npy")
ages = np.load("ages.npy")
genders = np.load("genders.npy")

X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    X,
    ages,
    genders,
    test_size=0.2,
    random_state=42
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)

x = Dense(256, activation="relu")(x)

x = Dropout(0.3)(x)

gender_output = Dense(
    1,
    activation="sigmoid",
    name="gender"
)(x)

age_output = Dense(
    1,
    activation="linear",
    name="age"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=[gender_output, age_output]
)

model.compile(
    optimizer="adam",
    loss={
        "gender":"binary_crossentropy",
        "age":"mse"
    },
    metrics={
        "gender":"accuracy",
        "age":"mae"
    }
)

history = model.fit(
    X_train,
    {
        "gender":gender_train,
        "age":age_train
    },
    validation_split=0.1,
    epochs=10,
    batch_size=32
)

model.save("models/age_gender_model.keras")

print("Training complete.")