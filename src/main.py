from queue import Queue
from WallEManager import WallEManager



def main():
    walle = WallEManager()
    walle.start()

    walle.loop()
    
        

if __name__ == '__main__':
    main()

    


    