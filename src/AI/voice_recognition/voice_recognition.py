from resemblyzer import VoiceEncoder, preprocess_wav
from AI.voice_recognition.micstream import MicStream
from AI.voice_recognition.voice_data import load_voice_embeddings, DEFAULT_SAVE_PATH
from queue import Queue
import numpy as np
import pyaudio
import resemblyzer
import speech_recognition as sr
import copy

# encoder = VoiceEncoder(device='cpu')

def voice_rec_loop(queue:list=None, savepath=DEFAULT_SAVE_PATH):
    vr = VoiceRecognition()
    vr.detect_speaker(queue=queue)


class VoiceRecognition:
    def __init__(self, savepath=DEFAULT_SAVE_PATH):
        self.encoder = VoiceEncoder(device='cpu')
        self.embeddings_path = savepath


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


    def get_speaker(self, audio_emb, speaker_embs:list, threshold=.8) -> int:
        '''
        Scans through list of embs and returns index of the speaker.
        '''
        for i, emb in enumerate(speaker_embs):
            if np.inner(emb, audio_emb) >= threshold:
                return i
        
        return -1



    def detect_speaker(self,
                    queue:list=None, 
                    chunk_count:int=20, 
                    threshold:float=.8,
                    voice_encodings_path=DEFAULT_SAVE_PATH) -> None:
        '''
        Runs a loop to listen to mic and detect who is speaking. Processes audio into strings that can be used.
        
        :param queue: Queue object that will be used to store the detected text and speaker. Default None to not store it.
        :param chunk_count: Int that represents how many chunks to sample to determine speaker. Default is 20 which is 2 seconds.
        :param threshold: threshold used to identify speaker similarity. Defaults to 0.8. Higher means more precise, smaller more leeway.
        
        '''

        voices = load_voice_embeddings(filename=voice_encodings_path)
        names, speaker_embs = [i[0] for i in voices], [i[1] for i in voices]

        audio = []
        command = []
        speakers = []   # used to find avg
        speaker = -1
        active_listening = False
        recognizer = sr.Recognizer()

        print('Now listening to microphone...')
        with MicStream(16000, 1600) as stream:
            audio_generator = stream.generator()
            for content in audio_generator:
                audio.append(content)

                if len(audio) > chunk_count:
                    audio.pop(0)

                # converts audio to wav and encodes the data.
                wav_bytes = b''.join(audio)

                wav = preprocess_wav(np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32))
                wav = resemblyzer.trim_long_silences(wav)
                data = self.encoder.embed_utterance(wav)
                speaker_index = self.get_speaker(data, speaker_embs, threshold=threshold)

                # if a voice is detected, add data into the full command array for when sample len > chunk_count
                # adds to the speaker to the list of who speaks at each sample to find avg.
                if active_listening:
                    command.append(content)
                    speakers.append(speaker_index)
                    # print(speakers)

                    # only uses the past chunk_count amount of data
                    if len(speakers) > chunk_count:
                        speakers.pop(0)
                
                # used to detect a new speaker and begin actively listening
                elif speaker_index >= 0:
                    print('detected speaker...')
                    command = copy.deepcopy(audio)
                    speakers = [speaker_index]
                    active_listening = True
                    speaker = speaker_index

                # passes audio data to be translated to text once it is detects the speaker is no longer speaking.
                # adds to shared queue if it is supplied in parameters
                if active_listening and speakers.count(speaker) / len(speakers) <= 0.33: # threshold for how often speaker should be talking
                    try:
                        transcript = recognizer.recognize_google(sr.AudioData(b''.join(command), 16000, pyaudio.get_sample_size(pyaudio.paInt16)))
                    
                    except sr.UnknownValueError:
                        transcript = 'An unknown error occured. Please try again.'
                    
                    finally:
                        active_listening = False

                    if queue is not None:
                        print('adding data to queue')
                        queue.append({
                            'speaker': names[speaker_index],
                            'speaker_index': speaker_index,
                            'transcript': transcript
                        })

                    else:
                        print(names[speaker_index], ': ', transcript)
            




if __name__ == '__main__':

    detect_speaker()

