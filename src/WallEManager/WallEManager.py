from AI.facial_recognition import facial_rec_loop, add_face, load_faces
from AI.voice_recognition import voice_rec_loop, add_voice
from multiprocessing import Process, Manager
from collections import defaultdict
from util import add_encoding
import sounddevice as sd
import time
from TTS.api import TTS
from num2words import num2words


def wait_for_que(que:list):
    while len(que) == 0:
        continue


class WallEManager:
    def __init__(self, processes:list=[]):
        '''
        Initializes a WallE bot manager to handle processes.

        :param processes: list of processes to instantiate. if empty initializes all.
        '''

        self.manager = Manager()
        self.data_ques = defaultdict(self.manager.list)
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        self.all_processes = defaultdict(None)

        self.process_map = {
            'face': {'func': facial_rec_loop, 'args': (), 'kwargs': {'que': self.data_ques['face'], 'namespace': self.setup_namespace()}},
            'voice': {'func': voice_rec_loop, 'args': (), 'kwargs': {'que': self.data_ques['voice'], 'namespace': self.setup_namespace()}}
        }

        if len(processes) == 0:
            processes = list(self.process_map.keys())

        for p in processes:
            if p in self.process_map.keys():
                d = self.process_map[p]
                self.all_processes[p] = Process(target=d['func'], args=d['args'], kwargs=d['kwargs'], daemon=True)


    def setup_namespace(self):
        '''
        Setups namespace with some default values
        '''
        res = self.manager.Namespace()
        res.reload = False

        return res

    
    def start(self):
        for name, p in self.all_processes.items():
            p.start()
            print(f'Started {name} process')


    def get_face(self):
        frque = self.data_ques['face']
        frque[:] = []

        self.speak('Now learning your face. Please make sure other people are not in the camera.')

        face_enc = None
        # loop to wait for face for 2 seconds before encoding it.
        while face_enc is None:
            start = time.time()
            while time.time() - start < 2:
                wait_for_que(frque)

                faces = frque.pop(0)
                # no faces detected
                if len(faces) != 1:
                    break

                name, loc, enc = faces[0]

                # # face is already known
                if name != 'Unknown':
                    print('face is already known.')
                    break
                
                if time.time() - start >= 2:
                    face_enc = enc
        
        return face_enc



    def get_mic_sample(self, secs=2) -> dict:
        '''
        Samples the mic for "secs" amount of seconds and gets the transcript from it.
        '''
        ns = self.process_map['voice']['kwargs']['namespace']
        que = self.data_ques['voice']
        que[:] = [] # clear que
        ns.sampletime = secs

        while len(que) == 0:
            continue

        data = que.pop(0)
        return data
        

    def learn_voice(self, name):
        
        que = self.data_ques['voice']
        que[:] = []

        self.speak('Please speak for about 5 seconds so that I can learn your voice.')

        while len(que) == 0:
            continue
    
        data = que.pop(0)
        # add_voice(data['audio_bytes'], name)

        self.speak('Thank you, ' + name)
        return data['audio_bytes']



    def add_person(self):
        '''
        Adds a new person to the known people. Learns their face and voice and saves it.
        '''
        face_enc = self.get_face()

        self.speak('Thank you. Please tell me your name.')
        name = None

        nsface = self.process_map['face']['kwargs']['namespace']
        nsvoice = self.process_map['voice']['kwargs']['namespace']

        while name is None:
            data = self.get_mic_sample()
            if data['success']:
                name = data['transcript']

        self.speak('Nice to meet you ' + data['transcript'])
        # add_face(name, enc)
        
        # learns voice
        audio_bytes = self.learn_voice(name)
        
        add_encoding(audio_bytes, face_enc, name)

        nsvoice.reload = True
        nsface.reload = True



    def join(self):
        '''
        waits for the processes to finish
        '''

        for _, p in self.all_processes:
            p.join()


    def speak(self, text, speaker='./AI/voice_recognition/test1.wav'):
        text_list = text.split()
        for i, t in enumerate(text_list):
            if t[0].isdigit():
                text_list[i] = num2words(t.replace(',', ''))

        wav = self.tts.tts(' '.join(text_list), speaker_wav='./AI/voice_recognition/test1.wav', language='en')
        sd.play(wav, 16000)
        sd.wait()


    
    def loop(self):
        '''
        main loop?
        '''
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

        while True:
            if len(self.data_ques['voice']) == 0:
                continue
            
            data = self.data_ques['voice'].pop(0)
            print(f'{data["speaker"]}: {data["transcript"]}')
            self.speak(data['transcript'])
            if data['transcript'].lower().strip('.') == 'add new person':
                self.add_person()

            
                

    
    
