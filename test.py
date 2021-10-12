import os
from efficientword.streams import SimpleMicStream
from efficientword.engine import HotwordDetector , MultiHotwordDetector
from efficientword import samples_loc

alexa_hw = HotwordDetector(
        hotword="Alexa",
        reference_file = os.path.join(samples_loc,"alexa_ref.json"),
    )

siri_hw = HotwordDetector(
        hotword="Siri",
        reference_file = os.path.join(samples_loc,"siri_ref.json")
    )

google_hw = HotwordDetector(
        hotword="Google",
        reference_file = os.path.join(samples_loc,"google_ref.json")
    )

multi_hw_engine = MultiHotwordDetector(
        detector_collection = [alexa_hw,siri_hw,google_hw]
    )

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Google / Alexa / Siri")
while True :
    frame = mic_stream.getFrame()
    result = multi_hw_engine.findBestMatch(frame)
    if(None not in result):
        print(result[0],f",Confidence {result[1]:0.4f}")

