from resemblyzer import VoiceEncoder, preprocess_wav
from AI.voice_recognition.micstream import MicStream
from AI.voice_recognition.voice_data import load_voice_embeddings
import numpy as np
import resemblyzer


encoder = VoiceEncoder(device='cpu')


def generate_speaker_embeddings(files:list):
    wavs = []
    for file in files:
        with open(file, 'rb') as tmp:
            wavs.append(preprocess_wav(np.frombuffer(tmp.read(), dtype=np.int16).astype(np.float32)))
    
    # return encoder.embed_speaker(wavs)
    wav = resemblyzer.trim_long_silences(wavs[0])
    return encoder.embed_utterance(wav)


def embed_audio(audio:bytes):

    with open(audio, 'rb') as audio:
        wav = preprocess_wav(np.frombuffer(audio.read(), dtype=np.int16).astype(np.float32))
    wav = resemblyzer.trim_long_silences(wav)
    embed = encoder.embed_utterance(wav)
    
    return embed


def get_speaker(audio_emb, speaker_embs:list, threshold=.8) -> int:
    '''
    Scans through list of embs and returns index of the speaker.
    '''
    for i, emb in enumerate(speaker_embs):
        if np.inner(emb, audio_emb) >= threshold:
            return i
    
    return -1



def detect_speaker(chunk_count=20, threshold=.8):
    '''Runs a loop to listen to mic and detect who is speaking. Chunk_count used to 
    determine how many chunks to process at a time.'''

    voices = load_voice_embeddings()
    names, speaker_embs = [i[0] for i in voices], [i[1] for i in voices]

    audio = []
    with MicStream(16000, 1600) as stream:
        audio_generator = stream.generator()
        for content in audio_generator:
            audio.append(content)

            if len(audio) > chunk_count:
                audio.pop(0)

            wav = preprocess_wav(np.frombuffer(b''.join(audio), dtype=np.int16).astype(np.float32))
            wav = resemblyzer.trim_long_silences(wav)
            data = encoder.embed_utterance(wav)
            
            speaker_index = get_speaker(data, speaker_embs, threshold=threshold)
            if speaker_index >= 0:
                print(names[speaker_index])
            else:
                print('No speaker detected.')



if __name__ == '__main__':

    detect_speaker()

