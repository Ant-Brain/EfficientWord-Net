from eff_word_net.streams import SimpleMicStream
import pyaudio
import librosa
import numpy as np

CHUNK = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
RECORD_SECONDS = 1.5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

def record_audio():
    inp_stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    frames = []
\
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = inp_stream.read(CHUNK)
        frames.append(data)

    inp_stream.stop_stream()
    inp_stream.close()

    return np.concatenate(frames)

def playFrame(inpFrame):
    print(inpFrame)
    converterFrame = librosa.resample(inpFrame, orig_sr=16000, target_sr=48000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=48000,
                         output=True,
                         #output_device_index=1
                         )
    #stream = p.open(format = p.get_format_from_width(1), channels = 1, rate = 16000, output = True)
    stream.write(converterFrame.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=1.5)

input("Press enter to record and wait for speak now:")

frame = record_audio()
input("Press Enter to play")
playFrame(frame)
frame = librosa.load("/home/captainamerica/Programming/EfficientWord-Net/EfficientWord-Net-Deployment/wakewords/alexa/alexa_en-GB_KateV3Voice.mp3",sr=16000)[0]
playFrame(frame) 