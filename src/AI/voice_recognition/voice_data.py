from resemblyzer import VoiceEncoder, preprocess_wav
from AI.voice_recognition.micstream import MicStream
import time
import pickle
import numpy as np


DEFAULT_SAVE_PATH = './data/encodings/voice_encodings'


def load_voice_embeddings(filename:str=DEFAULT_SAVE_PATH) -> list:
    '''loads the known voices from a file or returns empty array.'''

    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        print(e)
        return []
    

def save_voices(data:list, filename:str=DEFAULT_SAVE_PATH) -> None:
    '''
    Saves voice embeddings to a file.
    '''
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    except pickle.PicklingError as e:
        print(e)


def add_voice(sample: bytes, name: str) -> None:
    '''
    Adds a new voice from a sample of a voice as bytes.
    '''
    encoder = VoiceEncoder(device='cpu')

    data = np.frombuffer(sample, dtype=np.int16).astype(np.float32)
    data = preprocess_wav(data)

    emb = encoder.embed_utterance(data)

    current_embs = load_voice_embeddings()
    current_embs.append((name, emb))
    save_voices(current_embs)


def learn_voice(name:str, rec_time=10) -> bytes:
    '''
    Learns a new voice by sampling a 10 second audio stream
    from the mic and generating embeddings.

    Returns tuple with name and embeddings
    '''
    
    start = time.time()
    audio = []

    with MicStream() as stream:
        stream_generator = stream.generator()

        print(f'Recording audio for {rec_time} seconds...')
        for chunk in stream_generator:
            if time.time() - start > rec_time:
                break
            
            audio.append(chunk)
    
    print('Finished recording audio.')
    print('Generating embeddings...')
    #converts data into np array type float32
    data = np.frombuffer(b''.join(audio), dtype=np.int16).astype(np.float32)
    data = preprocess_wav(data)

    encoder = VoiceEncoder(device='cpu')
    emb = encoder.embed_utterance(data)

    return (name, emb)


if __name__ == '__main__':
    name, emb = learn_voice('Jarrod', rec_time=5)
    save_voices([(name, emb)])


