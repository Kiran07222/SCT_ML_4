# hand gesture recognition using CNN data set from Kaggle
# SkillCraft Technology Internship - Task 04
# Developed by: Kiran Vuyyalawada - 2024-06-15


import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
DATASET_PATH = r"C:\Users\nani3\Desktop\archive\leapGestRecog"
images = []
labels = []

print("Loading dataset...")

for person_folder in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_folder)

    if not os.path.isdir(person_path):
        continue

    for gesture_folder in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        label = gesture_folder

        for image_name in os.listdir(gesture_path):

            image_path = os.path.join(
                gesture_path,
                image_name
            )

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            images.append(img)
            labels.append(label)

print(f"Total Images Loaded: {len(images)}")

X = np.array(images, dtype="float32")
X = X / 255.0

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

num_classes = len(np.unique(y))

y = to_categorical(y, num_classes)

print("Classes Found:")
print(encoder.classes_)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))


model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(64, 64, 3)
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        num_classes,
        activation='softmax'
    )
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()



history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)


loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")

model.save("hand_gesture_model.keras")

print("\nModel Saved Successfully!")


with open("gesture_labels.txt", "w") as f:

    for label in encoder.classes_:
        f.write(label + "\n")

print("Labels Saved Successfully!")