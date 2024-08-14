import cv2
import face_recognition as fr
import face_data as fd
import numpy as np

STOP_FLAG = False
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
print('loading saved faces')
__ = fd.load_faces()
ENCODINGS, NAMES = [i[1] for i in __], [i[0] for i in __]


def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    face_locs = fr.face_locations(small_frame)
    face_encs = fr.face_encodings(small_frame, face_locs)
    # iterates over detected faces to give each a name
    names = []
    for enc in face_encs:
        name = 'Unknown'
        matches = fr.compare_faces(ENCODINGS, enc)
        if True in matches:
            # face_dist = fr.face_distance(ENCODINGS, enc)
            # smallest_dist = np.argmin(face_dist)
            # name = NAMES[smallest_dist]
            name = NAMES[matches.index(True)]
        
        names.append(name)

    # draws square around face with name
    for (top, right, bottom, left), name in zip(face_locs, names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def facial_rec_loop():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    waitval = 1
    missing = 1
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()

    while not STOP_FLAG and rval:
        cv2.imshow("preview", detect_face(frame))

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
        rval, frame = vc.read()
    
    cv2.destroyWindow("preview")
    vc.release()




if __name__ == '__main__':
    facial_rec_loop()
