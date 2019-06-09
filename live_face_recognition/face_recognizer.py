"""Recognize faces"""

import os
import face_recognition

FACESPATH = "/faces"

class FaceRecognizer():
    """Recognize faces"""

    def __init__(self):
        self.loadfaces()
        self.known_face_encodings = []
        self.known_face_names = []

    def loadfaces(self):
        """Load faces from the FACESPATH directory"""

        for file in os.listdir(FACESPATH):
            loaded_face = face_recognition.load_image_file(os.path.join(FACESPATH, file))
            face_encoding = face_recognition.face_encodings(loaded_face)[0]
            self.known_face_encodings.append(face_encoding)

            name = file.split('.')[0]
            self.known_face_names.append(name)
