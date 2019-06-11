"""Main"""

from . import face_detector
from . import face_recognizer
from . import liveness_detector
import cv2

class Main():
    """Main"""
    def __init__(self):
        self.fd = face_detector.FaceDetector()
        self.fr = face_recognizer.FaceRecognizer()
        self.ld = liveness_detector.LivenessDetector()
    
    def do_stuff(self):
        """Execute stuff"""

        video_capture = cv2.VideoCapture(0)

        progress = {}

        while True:
            _, frame = video_capture.read()

            image = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)[:, :, ::-1]

            face_locations = self.fd.find_faces(image)
            face_encodings = self.fd.get_face_encodings(frame, face_locations)
            face_names = self.fr.compare_faces(face_encodings)
            face_images = []

            for i in range(len(face_locations)):
                face_locations[i] = [x * 4 for x in face_locations[i]]
                top, right, bottom, left = face_locations[i]
                face_images.append(frame[top:bottom, left:right])

                preds, j, label = self.ld.detect_liveness(face_images[i])

                if label == 'real':
                    if face_names[i] not in progress.keys():
                        progress[face_names[i]] = 0
                    else:
                        progress[face_names[i]] += 1
                else:
                    progress[face_names[i]] = 0
                
                print (progress)
                
                if progress[face_names[i]] < 255:
                    if progress[face_names[i]] < 128:
                        color = (0, progress[face_names[i]] * 2, 255)
                    else:
                        color = (0, 255, 255 - (progress[face_names[i]] * 2))
                else:
                    color = (0, 255, 0)

                label = "{} - {}: {:.4f}".format(face_names[i], label, preds[j])


                cv2.putText(frame, label, (face_locations[i][0], face_locations[i][2] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (face_locations[i][0], face_locations[i][2]), (face_locations[i][1], face_locations[i][3]),
                    color, 2)



            cv2.imshow('face1', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
