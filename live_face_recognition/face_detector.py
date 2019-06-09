import face_recognition

class FaceDetector():
    def __init__(self, use_gpu_accelleration=False):
        self.use_gpu_accelleration = use_gpu_accelleration

    def findfaces(self, image):
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
