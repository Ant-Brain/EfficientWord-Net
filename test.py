import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net import samples_loc

mycroft_hw = HotwordDetector(
        hotword="Mycroft",
        reference_file = os.path.join(samples_loc,"mycroft_ref.json"),
    )

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Mycroft ")
while True :
    frame = mic_stream.getFrame()
    result = mycroft_hw.scoreFrame(frame)
    if result==None :
        #no voice activity
        continue
    if(result["match"]):
        print("Wakeword uttered",result["confidence"])
