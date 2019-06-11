"""Recognize faces"""

import os
import face_recognition
import numpy as np

FACESPATH = "./faces"

class FaceRecognizer():
    """Recognize faces"""

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_faces()

    def load_faces(self):
        """Load faces from the FACESPATH directory"""

        for file in os.listdir(FACESPATH):
            loaded_face = face_recognition.load_image_file(os.path.join(FACESPATH, file))
            face_encoding = face_recognition.face_encodings(loaded_face)[0]
            self.known_face_encodings.append(face_encoding)

            name = file.split('.')[0]
            self.known_face_names.append(name)

    def compare_faces(self, face_encodings):
        """Compare face encodings with known face encodings"""

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_names
