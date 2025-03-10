import pyaudio
from typing import Callable
import numpy as np
from scipy.signal import resample  # For downsampling

NoParameterFunction = Callable[[], None]
AudioFrameFunction = Callable[[], np.array]

class CustomAudioStream:
    """
    CustomAudioStream applies a sliding window to an audio stream.
    """
    def __init__(
        self,
        open_stream: Callable[[], None],
        close_stream: Callable[[], None],
        get_next_frame: Callable[[], np.array],
        window_length_secs=1,
        sliding_window_secs: float = 1/8,
        sample_rate=16000  # Target sample rate for the engine
    ):
        self._open_stream = open_stream
        self._close_stream = close_stream
        self._get_next_frame = get_next_frame
        self._sample_rate = sample_rate
        self._window_size = int(window_length_secs * sample_rate)
        self._sliding_window_size = int(sliding_window_secs * sample_rate)
        self._out_audio = np.zeros(self._window_size)  # Initialize audio buffer
        print("Initial output buffer shape:", self._out_audio.shape)

    def start_stream(self):
        self._out_audio = np.zeros(self._window_size)
        self._open_stream()
        for i in range(self._sample_rate // self._sliding_window_size - 1):
            self.getFrame()

    def close_stream(self):
        self._close_stream()
        self._out_audio = np.zeros(self._window_size)

    def getFrame(self):
        """
        Returns a 1-second audio frame with a sliding window of length (sliding_window_secs)
        using the target sample rate.
        """
        new_frame = self._get_next_frame()
        assert new_frame.shape == (self._sliding_window_size,), \
            "audio frame size from src doesn't match sliding_window_secs"
        self._out_audio = np.append(
            self._out_audio[self._sliding_window_size:],
            new_frame 
        )
        return self._out_audio

class SimpleMicStream(CustomAudioStream):
    def __init__(self, window_length_secs=1, sliding_window_secs: float = 1/8,
                 custom_channels=2, custom_rate=48000, custom_device_index=None):
        p = pyaudio.PyAudio()
        # Calculate CHUNK based on sliding window seconds and capture rate (custom_rate)
        CHUNK = int(sliding_window_secs * custom_rate)
        print("Chunk size (captured at {}Hz): {}".format(custom_rate, CHUNK))
        
        mic_stream = p.open(
            format=pyaudio.paInt16,
            channels=custom_channels,
            rate=custom_rate,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=custom_device_index
        )
        mic_stream.stop_stream()

        def get_next_frame():
            try:
                # Use exception_on_overflow=False to avoid overflow errors
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print("Input overflow:", e)
                # Return a silent frame if overflow occurs
                data = b'\x00' * CHUNK * 2 * custom_channels  # 2 bytes per sample
            arr = np.frombuffer(data, dtype=np.int16)
            if custom_channels > 1:
                # Convert stereo to mono by averaging channels
                arr = np.mean(arr.reshape(-1, custom_channels), axis=1).astype(np.int16)
            # Downsample from custom_rate (48000Hz) to target rate (16000Hz)
            target_rate = 16000
            new_length = int(len(arr) * target_rate / custom_rate)
            arr_down = resample(arr, new_length).astype(np.int16)
            return arr_down

        # Initialize the CustomAudioStream with the target sample rate for the engine
        CustomAudioStream.__init__(
            self,
            open_stream=mic_stream.start_stream,
            close_stream=mic_stream.stop_stream,
            get_next_frame=get_next_frame,
            window_length_secs=window_length_secs,
            sliding_window_secs=sliding_window_secs,
            sample_rate=16000  # Engine expects 16000 Hz
        )
