"""Detect faces"""
import face_recognition

class FaceDetector():
    """Detect faces"""
    def __init__(self, use_gpu_accelleration=False):
        self.use_gpu_accelleration = use_gpu_accelleration

    def find_faces(self, image):
        """Find all face locations in the provided image, returns a list of face locations"""
        face_locations = []

        if self.use_gpu_accelleration:
            face_locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=0,
                model="cnn"
            )
        else:
            face_locations = face_recognition.face_locations(image)

        # face_images = []

        # for face_loc in face_locations:
        #     top, right, bottom ,left = face_loc
        #     face_images.append(image[top:bottom, left:right])

        return face_locations

    def get_face_encodings(self, image, face_locations):
        """Get face encodings from face locations on image"""
        return face_recognition.face_encodings(image, face_locations)