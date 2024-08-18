from resemblyzer import VoiceEncoder, preprocess_wav
from AI.voice_recognition.micstream import MicStream
import time
import pickle
import numpy as np


def load_voice_embeddings(filename:str='voice_embeddings') -> list:
    '''loads the known voices from a file or returns empty array.'''

    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        print(e)
        return []
    

def save_voices(data:list, filename:str='voice_embeddings') -> None:
    '''
    Saves voice embeddings to a file.
    '''
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    except pickle.PicklingError as e:
        print(e)


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


