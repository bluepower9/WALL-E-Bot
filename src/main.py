from AI.voice_recognition import detect_speaker
from queue import Queue
from multiprocessing import Manager, Process
from WallEManager import WallEManager


def main():
    walle = WallEManager()
    walle.start()

    walle.loop()
    
        

if __name__ == '__main__':
    main()