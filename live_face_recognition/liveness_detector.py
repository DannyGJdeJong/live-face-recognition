"""Detect liveness"""
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np

class LivenessDetector():
    """"Detect liveness"""

    def __init__(self):
        self.load_model()

    def load_model(self):
        """Loads the liveness model"""
        self.model = load_model("liveness.model")

    def detect_liveness(self, face_image):
        """Detects liveness"""
        face_image = cv2.resize(face_image, (32, 32))
        face_image = face_image.astype("float") / 255.0
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)

        prediction = self.model.predict(face_image)[0]
        j = np.argmax(prediction)
        label = ['real', 'fake'][j]
        return prediction, j, label
