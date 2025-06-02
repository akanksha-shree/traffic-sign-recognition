import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Class labels (optional: replace with actual sign names if available)
classes = [str(i) for i in range(43)]

# Load and preprocess test image
img_path = 'Test/00001.png'  # Change this to your test image path
image = cv2.imread(img_path)
image = cv2.resize(image, (30, 30))
image = np.expand_dims(image, axis=0)
image = image / 255.0

# Predict
pred = model.predict(image)
class_id = np.argmax(pred)
confidence = np.max(pred)

print(f"Predicted Class: {class_id} (Confidence: {confidence*100:.2f}%)")
