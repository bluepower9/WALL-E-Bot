from AI.facial_recognition import facial_rec_loop
from AI.voice_recognition import voice_rec_loop
from multiprocessing import Process, Manager
from collections import defaultdict
import sounddevice as sd
from TTS.api import TTS
from num2words import num2words


class WallEManager:
    def __init__(self, processes:list=[]):
        '''
        Initializes a WallE bot manager to handle processes.

        :param processes: list of processes to instantiate. if empty initializes all.
        '''

        self.manager = Manager()
        self.data_ques = defaultdict(self.manager.list)
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        self.all_processes = []

        self.process_map = {
            'face': {'func': facial_rec_loop, 'args': (), 'kwargs': {}},
            'voice': {'func': voice_rec_loop, 'args': (), 'kwargs': {'queue': self.data_ques['voice']}}
        }

        if len(processes) == 0:
            processes = list(self.process_map.keys())

        for p in processes:
            if p in self.process_map.keys():
                d = self.process_map[p]
                self.all_processes.append((p, Process(target=d['func'], args=d['args'], kwargs=d['kwargs'], daemon=True)))

    
    def start(self):
        for name, p in self.all_processes:
            p.start()
            print(f'Started {name} process')


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

            
                

    
    
