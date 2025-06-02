import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Constants
DATA_DIR = 'Train/'
IMG_HEIGHT, IMG_WIDTH = 30, 30
NUM_CLASSES = 43

# Load data
X = []
Y = []

print("Loading and preprocessing images...")
for class_id in range(NUM_CLASSES):
    path = os.path.join(DATA_DIR, str(class_id))
    images = os.listdir(path)
    for img in images:
        try:
            image = cv2.imread(os.path.join(path, img))
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            X.append(image)
            Y.append(class_id)
        except:
            print("Error loading image:", img)

X = np.array(X)
Y = np.array(Y)

# Normalize and one-hot encode
X = X / 255.0
Y = to_categorical(Y, NUM_CLASSES)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_val, y_val))

model.save("model.h5")
print("Model saved as model.h5")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
