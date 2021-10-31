import pyaudio
from typing import Tuple , Callable
import numpy as np
from eff_word_net.engine import HotwordDetector
from eff_word_net import RATE

NoParameterFunction = Callable[[],None]
AudioFrameFunction = Callable[[],np.array]


class CustomAudioStream :
    """
    CustomAudioStream implementation allows developers to use 
    any 16000Hz sampled audio streams with inference engine

    It tries to add sliding window to audio streams
    """
    def __init__(
        self,
        open_stream:Callable[[],None],
        close_stream:Callable[[],None],
        get_next_frame:Callable[[],np.array],
        sliding_window_secs:float = 1/8
        ):

        self._out_audio = np.zeros(RATE) #blank 1 sec audio

        self._open_stream = open_stream
        self._close_stream = close_stream
        self._get_next_frame = get_next_frame
        self._sliding_window_size = int(sliding_window_secs * RATE)

    def start_stream(self):
        self._out_audio = np.zeros(RATE)
        self._open_stream()
        for i in range(RATE//self._sliding_window_size -1):
            self.getFrame()

    def close_stream(self):
        self._close_stream()
        self._out_audio = np.zeros(RATE)

    def getFrame(self):
        """
        Returns a 1 sec audio frame with sliding window of 1/8 sec with 
        sampling frequency 16000Hz
        """

        new_frame = self._get_next_frame()
        assert new_frame.shape == (self._sliding_window_size,), \
            "audio frame size from src doesnt match sliding_window_secs"

        self._out_audio = np.append(
                self._out_audio[self._sliding_window_size:],
            new_frame 
        )

        return self._out_audio

class SimpleMicStream(CustomAudioStream) :

    """
    Implements mic stream with sliding window, 
    implemented by inheriting CustomAudioStream
    """
    def __init__(self,sliding_window_secs:float=1/8):
        p=pyaudio.PyAudio()

        CHUNK = int(sliding_window_secs*RATE)

        mic_stream=p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=CHUNK
        )
        mic_stream.stop_stream()

        CustomAudioStream.__init__(
            self,
            open_stream = mic_stream.start_stream,
            close_stream = mic_stream.stop_stream,
            get_next_frame = lambda : (
                np.frombuffer(mic_stream.read(CHUNK),dtype=np.int16)
                ),
        )
