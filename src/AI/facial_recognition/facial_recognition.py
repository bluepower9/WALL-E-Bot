from util import load_people_encodings, ENCODINGS_FILEPATH
import cv2
import face_recognition as fr
import AI.facial_recognition.face_data as fd


def facial_rec_loop(que=None, namespace=None):
    fr = FacialRecognition(que=que, namespace=namespace)
    fr.start()


class FacialRecognition:
    def __init__(self, que:list=None, namespace=None, save_path=ENCODINGS_FILEPATH):
        self.STOP_FLAG = False
        self.que = que
        self.namespace = namespace
        self.save_path = save_path
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        self.load_face_encodings()


    def load_face_encodings(self):
        print('loading saved faces')
        filepath = self.save_path
        if self.save_path is None:
            filepath = self.EMBEDDINGS_PATH

        self.encodings = load_people_encodings(filename=filepath)
        self.known_faces, self.userids = [], []

        for k, v in self.encodings.items():
            del v['voice_encoding']
            self.known_faces.append(v['face_encoding'])
            self.userids.append(k)

        
    def detect_face(self, frame):
        '''
        detects face from a given frame. Converts color scheme from BGR to RGB and 
        scales image down to 50%. Encodes face and compares it to known faces to
        determine who the detected face is. Lists as "unknown" if not known.
        '''
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

        face_locs = fr.face_locations(small_frame)
        face_encs = fr.face_encodings(small_frame, face_locs)
        # iterates over detected faces to give each a name
        names = []

        for enc in face_encs:
            name = 'Unknown'
            matches = fr.compare_faces(self.known_faces, enc)

            if True in matches:
                # face_dist = fr.face_distance(ENCODINGS, enc)
                # smallest_dist = np.argmin(face_dist)
                # name = NAMES[smallest_dist]

                id = self.userids[matches.index(True)]   # gets the first match and returns that name
                name = self.encodings[id]['name']
            
            names.append(name)
        
        # adds to data que if faces found
        # if self.que is not None and len(names) > 0:
        self.que.append(list(zip(names, face_locs, face_encs, strict=True)))

        # only holds 50 frames of data to prevent too large of list
        if len(self.que) > 50:
            self.que.pop(0)

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


    def start(self):
        '''
        Main loop to run facial recognition via webcam.
        '''
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()

        while not self.STOP_FLAG and rval:
            if self.namespace.reload:
                self.load_face_encodings()
                self.namespace.reload = False

            cv2.imshow("preview", self.detect_face(frame))

            key = cv2.waitKey(10)
            if key == 27: # exit on ESC
                break
            rval, frame = vc.read()
        
        cv2.destroyWindow("preview")
        vc.release()


