import os
from efficientword.streams import SimpleMicStream
from efficientword.engine import HotwordDetector , MultiHotwordDetector
from efficientword import samples_loc

alexa_hw = HotwordDetector(
        hotword="Alexa",
        reference_file = os.path.join(samples_loc,"alexa_ref.json"),
    )

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Alexa ")
while True :
    frame = mic_stream.getFrame()
    result = alexa_hw.checkFrame(frame)
    if(result):
        print("Wakeword uttered")

