#!.venv/bin/python
import cv2
from live_face_recognition import face_detector
from live_face_recognition import liveness_detector




ld = liveness_detector.LivenessDetector()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    fd = face_detector.FaceDetector()
    face_locations = fd.findfaces(frame)

    face_images = []

    for face_loc in face_locations:
        top, right, bottom ,left = face_loc
        face_images.append(frame[top:bottom, left:right])

    for i in range(len(face_locations)):
        preds, j, label = ld.detect_liveness(face_images[i])
        label = "{}: {:.4f}".format(label, preds[j])
        cv2.putText(frame, label, (face_locations[i][0], face_locations[i][2] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (face_locations[i][0], face_locations[i][2]), (face_locations[i][1], face_locations[i][3]),
            (0, 0, 255), 2)


    cv2.imshow('face1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
