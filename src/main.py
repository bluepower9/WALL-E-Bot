from queue import Queue
from multiprocessing import Manager, Process
from WallEManager import WallEManager
from TTS.api import TTS
import sounddevice as sd



def main():
    walle = WallEManager()
    walle.start()

    walle.loop()
    
        

if __name__ == '__main__':
    main()

    


    