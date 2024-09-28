from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import json
import pickle

ENCODINGS_FILEPATH = './data/encodings/people_encodings'

def read_configs(filename='config.json') -> dict:
    data = None
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

    except Exception as e:
        print(f'Error reading config file: {filename} - Exception: {str(e)}')
    
    return data


def load_people_encodings(filename=ENCODINGS_FILEPATH):
    '''
    loads and returns the encodings for people.

    format:
    {
        userid (int): {
            name (str),
            face_encoding (tensor),
            voice_encoding (tensor)
        }
    }
    '''
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return data

    except (FileNotFoundError, pickle.PicklingError) as e:
        print(e)
        return {}
        



def add_encoding(audio_bytes, face_enc, name, filename=ENCODINGS_FILEPATH):
    encodings = load_people_encodings(filename=filename)
    userid = 0
    
    if len(encodings) > 0:
        userid = max(encodings, key=lambda x: x.keys()) + 1
    
    encoder = VoiceEncoder(device='cpu')
    data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    data = preprocess_wav(data)

    voice_enc = encoder.embed_utterance(data)


    encodings[userid] = {
        'name': name,
        'face_encoding': face_enc,
        'voice_encoding': voice_enc
    }

    try:
        with open(filename, 'wb') as file:
            pickle.dump(encodings, file)

    except pickle.PicklingError as e:
        print(e)
        

