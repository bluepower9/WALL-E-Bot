import face_recognition as fr
import pickle

def load_faces(path='./face_encodings'):
    try:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        print(e)
        return []


def save_faces(faces, path='./face_encodings'):
    try:
        with open(path, 'wb') as file:
            pickle.dump(faces, file)
    except pickle.PicklingError as e:
        print(e)
    

def add_face(name, face_enc, path='./face_encodings'):
    faces = load_faces(path=path)
    faces.append((name, face_enc))
    save_faces(faces, path=path)

    return faces


if __name__ == '__main__':
    img = fr.load_image_file('test.jpg')
    img_enc = fr.face_encodings(img)[0]

    faces = add_face('Jarrod', img_enc)
    print(faces)
