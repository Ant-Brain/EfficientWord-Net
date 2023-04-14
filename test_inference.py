import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector, MultiHotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

base_model = Resnet50_Arc_loss()

mycroft_hw = HotwordDetector(
    hotword="mycroft",
    model = base_model,
    reference_file="mycroft_ref.json",
    threshold=0.7,
    relaxation_time=2
)

alexa_hw = HotwordDetector(
        hotword="alexa",
        model=base_model,
        reference_file = "alexa_ref.json",
        threshold=0.7,
        relaxation_time=2,
        #verbose=True
)

balloon_hw = HotwordDetector(
    hotword="balloon",
    model=base_model,
    reference_file="balloon_ref.json",
    threshold=0.65,
    relaxation_time=2,
    #verbose=True
)

computer_hw = HotwordDetector(
    hotword="computer",
    model=base_model,
    reference_file="computer_ref.json",
    threshold=0.7,
    relaxation_time=2,
    #verbose=True
)

mobile_hw = HotwordDetector(
    hotword="mobile",
    model = base_model,
    reference_file="mobile_ref.json",
    threshold=0.7,
    relaxation_time=2,
    #verbose=True
)

lights_on = HotwordDetector(
    hotword="lights_on",
    model = base_model,
    reference_file="lights_on_ref.json",
    threshold=0.7,
    relaxation_time=2    
)


lights_off = HotwordDetector(
    hotword="lights_on",
    model = base_model,
    reference_file="lights_off_ref.json",
    threshold=0.7,
    relaxation_time=2    
)

multi_hotword_detector = MultiHotwordDetector(
    [mycroft_hw, alexa_hw, balloon_hw, computer_hw, mobile_hw, lights_on, lights_off],
    model=base_model,
    continuous=True,
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75
)
mic_stream.start_stream()

#print("Say ", mycroft_hw.hotword)
print("Say one hotword among :", " ".join([x.hotword for x in multi_hotword_detector.detector_collection]))
while True :
    frame = mic_stream.getFrame()
    best_match = multi_hotword_detector.findBestMatch(frame)
    if best_match[0]!=None :
        print(best_match[0].hotword, best_match[1])
    