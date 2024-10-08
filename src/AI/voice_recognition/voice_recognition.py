from resemblyzer import VoiceEncoder, preprocess_wav, trim_long_silences
from AI.voice_recognition.micstream import MicStream
from AI.voice_recognition.voice_data import load_voice_embeddings
from util import load_people_encodings, ENCODINGS_FILEPATH
import numpy as np
import pyaudio
import resemblyzer
import speech_recognition as sr
import copy
import time
import audioop as ap
from typing import Union


def voice_rec_loop(que:list=None, namespace=None, savepath=ENCODINGS_FILEPATH):
    '''
    Function to create and run VoiceRecognition() loop. Useful for multiprocessing.

    :param que: Multiprocessing Manager list used as que for data transfer to host process.
    :param namespace: Namespace object to set flags for the process.
    :param savepath: Savepath for the voice encodings.
    '''
    vr = VoiceRecognition(namespace=namespace, que=que)
    vr.detect_speaker()


class VoiceRecognition:
    def __init__(self, namespace=None, savepath=ENCODINGS_FILEPATH, que:list=None):
        self.namespace = namespace
        self.que = que
        self.setup_namespace()
        self.encoder = VoiceEncoder(device='cpu')
        self.EMBEDDINGS_PATH = savepath
        self.MIN_THRESHOLD = 200
        self.threshold = self.MIN_THRESHOLD # minimum energy threshold
        self.load_embeddings(savepath)        


    def load_embeddings(self, filepath=None):
        if filepath is None:
            filepath = self.EMBEDDINGS_PATH

        self.encodings = load_people_encodings(filename=filepath)
        for k, v in self.encodings.items():
            del v['face_encoding']


    def setup_namespace(self):
        if self.namespace is not None:
            self.namespace.sampletime = 0


    def generate_speaker_embeddings(self, files:list):
        wavs = []
        for file in files:
            with open(file, 'rb') as tmp:
                wavs.append(preprocess_wav(np.frombuffer(tmp.read(), dtype=np.int16).astype(np.float32)))
        
        # return encoder.embed_speaker(wavs)
        wav = resemblyzer.trim_long_silences(wavs[0])
        return self.encoder.embed_utterance(wav)


    def embed_audio(self, audio:bytes):

        with open(audio, 'rb') as audio:
            wav = preprocess_wav(np.frombuffer(audio.read(), dtype=np.int16).astype(np.float32))
        wav = resemblyzer.trim_long_silences(wav)
        embed = self.encoder.embed_utterance(wav)
        
        return embed


    def get_speaker(self, audio_emb: Union[list,bytes], threshold=.7) -> int:
        '''
        Scans through list of embs and returns index of the speaker.

        :param audio_emb: if a list is given, finds the average speaker found and returns it. Otherwise, processes the whole sample at once.
        :param speaker_embs: list of speakers and their embeddings. see voice_data.py for loading and saving.
        :param threshold: the threshold value (0-1) used to determine how similar the audio is. A higher value means more precise.

        :return speaker_index: Returns the speaker index from the list of speaker_embs
        '''
        if type(audio_emb) == list:
            audio_emb = b''.join(audio_emb)
        
        index, maxscore = -1, 0
        wav = preprocess_wav(np.frombuffer(audio_emb, dtype=np.int16).astype(np.float32))
        wav = trim_long_silences(wav)
        data = self.encoder.embed_utterance(wav)

        for i, v in self.encodings.items():
            emb = v['voice_encoding']
            score = np.inner(emb, data)
            print('speaker score: ', score)
            if score >= threshold and score > maxscore:
                index = i
                maxscore = score
 
        return index


    def set_ambient_threshold(self, stream:MicStream, secs=2, energy_ratio=1.5) -> None:
        '''
        Samples the audio stream to determine an audio threshold.
        '''
        print('Listening to to ambient noise to set noise threshold...')
        gen = stream.generator()
        chunk_count = int(secs * 10)
        sr.Recognizer().adjust_for_ambient_noise
        audio = []

        for i in range(chunk_count):
            audio.append(ap.rms(next(gen), stream.WIDTH))
        
        self.threshold = max(self.MIN_THRESHOLD, int(sum(audio)/len(audio) * energy_ratio))
        print('Set threshold to: ', self.threshold)



    def get_phrase(self, stream, timeout=None) -> list:
        '''
        Automatically waits and returns a full phrase retrieved from mic. Waits for the energy threshold to pass
        the set threshold and then listens until the energy goes back down below the threshold.
        
        '''
        active_listen = False
        sample = [] # samples in 1 second intervals of chunks (10 chunks)
        phrase = []
        energylvls = []

        generator = stream.generator()

        for chunk in generator:
            sample.append(chunk)

            if len(sample) > 10:
                sample.pop(0)

            energy = ap.rms(b''.join(sample), 2)

            if active_listen:
                phrase.append(chunk)
                energylvls.append(energy)
                if len(energylvls) > 10:
                    energylvls.pop(0)

            # sets active listen state if energy > threshold for the 1 sec sample
            if not active_listen and energy > self.threshold:
                print('Detected speaker')
                active_listen = True
                phrase = copy.deepcopy(sample)
                energylvls = [energy]

            # returns when actively listening and energy falls below threshold and for certain amount of time.
            if active_listen and sum(energylvls)/len(energylvls) < self.threshold:
                return phrase  


    def detect_speaker(self, similarity_threshold:float=.8) -> None:
        '''
        Runs a loop to listen to mic and detect who is speaking. Processes audio into strings that can be used.
        
        :param similarity_threshold: threshold used to identify speaker similarity. Defaults to 0.8. Higher means more precise, smaller more leeway.
        
        '''
        recognizer = sr.Recognizer()    

        print('Now listening to microphone...')
        with MicStream(16000, 1600) as stream:
            self.set_ambient_threshold(stream)

            while True:
                phrase = self.get_phrase(stream)

                if self.namespace.reload:
                    print('reloading embeddings')
                    self.load_embeddings()
                    self.namespace.reload = False


                wav_bytes = b''.join(phrase)
                speaker_index = self.get_speaker(phrase, threshold=similarity_threshold)
                
                try:
                    audiodata = sr.AudioData(wav_bytes, 16000, pyaudio.get_sample_size(pyaudio.paInt16))
                    # transcript = recognizer.recognize_sphinx(audiodata, keyword_entries=[('wally', 1e-20)])
                    transcript = recognizer.recognize_google(audiodata)
                    success = True
                    # print('transcript sphinx: ', transcript)
                    print('google transcript: ', transcript)
                
                except sr.UnknownValueError:
                    transcript = 'An unknown error occured. Please try again.'
                    success = False


                if self.que is not None:
                    self.que.append({
                        'speaker': self.encodings[speaker_index]['name'] if speaker_index >= 0 else 'Unknown',
                        'speaker_index': speaker_index,
                        'transcript': transcript,
                        'audio_bytes': wav_bytes,
                        'start_time': time.time(),
                        'success': success
                    })


