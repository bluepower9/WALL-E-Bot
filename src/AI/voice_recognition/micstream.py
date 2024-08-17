from pyaudio import PyAudio
import pyaudio
import queue


class MicStream:
    def __init__(self, rate=16000, chunk=1600, channels=1):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.closed = False

        self._buf = queue.Queue()

    
    def __enter__(self):
        '''Creates PyAudio instances and opens a stream to get mic audio'''

        self.audio = PyAudio()
        self.stream = self.audio.open(
            input = True,
            format = pyaudio.paInt16,
            rate = self.rate,
            frames_per_buffer = self.chunk,
            channels = self.channels,
            stream_callback = self.__add_to_buffer
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        '''Closes streams and PyAudio instances'''

        self.stream.stop_stream()
        self.stream.close()
        self._buf.put(None)
        self.audio.terminate()
        self.closed = True

    
    def __add_to_buffer(self, in_data, frame_count, time_info, status_flags):
        '''Used to asynchronously add frames to buffer'''

        self._buf.put(in_data)
        return None, pyaudio.paContinue
    

    def generator(self):

        while not self.closed:
            frame = self._buf.get()
            if frame is None:
                return
            
            data = [frame]

            while True:
                try:
                    frame = self._buf.get(block=False)
                    if frame is None:
                        return
                    data.append(frame)
                except queue.Empty:
                    break

            yield b"".join(data)


    

    

    
        