from AI.facial_recognition import facial_rec_loop
from AI.voice_recognition import detect_speaker
from multiprocessing import Process, Manager


class WallEManager:
    def __init__(self, processes:list=[]):
        '''
        Initializes a WallE bot manager to handle processes.

        :param processes: list of processes to instantiate. if empty initializes all.
        '''

        self.manager = Manager()
        self.que = self.manager.list()
        self.all_processes = []

        self.process_map = {
            'face': {'func': facial_rec_loop, 'args': (), 'kwargs': {}},
            'voice': {'func': detect_speaker, 'args': (), 'kwargs': {'queue': self.que}}
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

    
    def loop(self):
        '''
        main loop?
        '''
        while True:
            if len(self.que) == 0:
                continue
            
            data = self.que.pop(0)
            print(f'{data["speaker"]}: {data["transcript"]}')
        

    
    
