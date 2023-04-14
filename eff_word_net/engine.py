import json
from os.path import isfile , join
import numpy as np
import pyaudio

from typing import Tuple , List ,Union

from eff_word_net.audio_processing import First_Iteration_Siamese, ModelRawBackend, Resnet50_Arc_loss
from eff_word_net import RATE
from time import time as current_time_in_sec
import logging

from eff_word_net.audio_processing import MODEL_TYPE_MAPPER

class HotwordDetector :

    """
    EfficientWord based HotwordDetector Engine implementation class
    """

    def __init__(
            self,
            hotword:str,
            model:ModelRawBackend,
            reference_file:str,
            threshold:float=0.9,
            relaxation_time=0.8,
            continuous=True,
            verbose = False):
        """
        Intializes hotword detector instance

        Inp Parameters:

            hotword : hotword in a string

            model : model to be used
            
            reference_file : path of reference file for a hotword generated 
            with efficientword.generate_reference module

            threshold: float value between 0 and 1 , min similarity score
            required for a match

            relaxation_time : the detector uses a sliding window approach to check for triggers, 
            which results in multiple triggers per utterance. This parameter mentions the relaxation_time for the next trigger

            continuous: bool value to know if a HotwordDetector is operating on a single continuous stream , else false

        """
        assert isfile(reference_file), \
            "Reference File Path Invalid"

        assert threshold>0 and threshold<1, \
            "Threshold can be only between 0 and 1"

        data = json.loads(open(reference_file,'r').read())
        self.embeddings = np.array(data["embeddings"]).astype(np.float32)

        #self.model_type = data.get()
        assert self.embeddings.shape[0]>3, \
            "Minimum of 4 sample datapoints is required"
        
        assert MODEL_TYPE_MAPPER[data["model_type"]]==type(model)
        self.model = model

        self.hotword = hotword
        self.threshold = threshold
        self.continuous = continuous

        self.relaxation_time = relaxation_time
        self.verbose = verbose

        self.__last_activation_time = current_time_in_sec()

    def __repr__(self):
        return f"Hotword: {self.hotword}"

    def __crossedRelaxationTime(self):
        current_time = current_time_in_sec()
        print("gap :",current_time - self.__last_activation_time)
        return (current_time-self.__last_activation_time) > self.relaxation_time

    def scoreVector(self,inp_vec:np.array) -> float :
        
        score =  self.model.scoreVector(inp_vec, self.embeddings)
        current_time = current_time_in_sec()

        if self.continuous :
            if score>self.threshold :
                if ( current_time - self.__last_activation_time ) < self.relaxation_time :
                    return 0.001
        
        if score>self.threshold :
            if self.verbose : 
                print(f"Activation Gap for {self.hotword}:", current_time - self.__last_activation_time)
            self.__last_activation_time = current_time
        
        return score
          
    def scoreFrame(
            self,
            inp_audio_frame:np.array,
            unsafe:bool = False) -> float :
        """
        Converts given audio frame to embedding and checks for similarity
        with given reference file

        Inp Parameters:

            inp_audio_frame : np.array of 1channel 1 sec 16000Hz sampled audio 
            frame
            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech or silence, to minimalize false positives

        **Note : change unsafe to True only if you know what you are doing**

        Out Parameters:

            {
                "match":True or False,
                "confidence":float value
            }
                 or 
            None when no voice activity is identified
        """

        if(not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:RATE//10]
            )
            if(upperPoint > 0.2):
                return None

        #assert inp_audio_frame.shape == (RATE,), \
        #    f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        score = self.scoreVector(
            self.model.audioToVector(
                inp_audio_frame
            )
        )

        return {
            "match":score >= self.threshold,
            "confidence":score
        }

HotwordDetectorArray = List[HotwordDetector]
MatchInfo = Tuple[HotwordDetector,float]
MatchInfoArray = List[MatchInfo]

class MultiHotwordDetector :
    """
    Wrapper over HotwordDetector to check for presence of one out of
    multiple hotwords efficiently
    """

    def __init__(
        self,
        detector_collection:HotwordDetectorArray,
        model:ModelRawBackend,
        continuous=True
    ):
        """
        Inp Parameters:

            detector_collection : List/Tuple of HotwordDetector instances
        """
        assert len(detector_collection)>1, \
            "Pass atleast 2 HotwordDetector instances"

        for detector in detector_collection :
            assert isinstance(detector,HotwordDetector), \
                "Mixed Array received, send HotwordDetector only array"

        self.model = model

        self.detector_collection = detector_collection
        self.continous = continuous

    def findBestMatch(
            self,
            inp_audio_frame:np.array,
            unsafe:bool=False
            ) -> MatchInfo :
        """
        Returns the best match hotword for a given audio frame
        within respective thresholds , returns None if found none

        Inp Parameters:

            inp_audio_frame : 1 sec 16000Hz frq sampled audio frame 

            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech , to minimalize false positives

        **Note : change unsafe to True only if you know what you are doing**

        Out Parameters:

            (detector,score) : returns detector of best matched hotword ,
            with its score

        """
        #assert inp_audio_frame.shape == (RATE,), \
        #    f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        """
        if(not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:1600]
            )
            if(upperPoint > 0.2):
                return None , None
        """
        embedding = self.model.audioToVector(inp_audio_frame)

        best_match_detector:str = None
        best_match_score:float = 0.0

        for detector in self.detector_collection :
            score = detector.scoreVector(embedding)

            if(score < detector.threshold):
                continue

            if(score>best_match_score):
                best_match_score = score
                best_match_detector = detector
        return (best_match_detector,best_match_score)

    def findAllMatches(
            self,
            inp_audio_frame:np.array,
            unsafe:bool=False
            ) -> MatchInfoArray :
        """
        Returns the best match hotword for a given audio frame
        within respective thresholds , returns None if found none

        Inp Parameters:

            inp_audio_frame : 1 sec 16000Hz frq sampled audio frame

            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech , to minimalize false positives

        Note : change unsafe to True only if you know what you are doing

        Out Parameters:

            [ (detector,score) ,... ] : returns list of matched detectors 
            with respective scores

        """
        #assert inp_audio_frame.shape == (RATE,), \
        #    f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"


        if self.continous and (not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:1600]
            )
            if(upperPoint > 0.2 or upperPoint==0):
                return None , None

        embedding = self.model.audioToVector(inp_audio_frame)

        matches:MatchInfoArray = []

        best_match_score = 0.0
        for detector in self.detector_collection :
            score = detector.getMatchScoreVector(embedding)
            print(detector,score,end="|")
            if(score<detector.threshold):
                continue
            if(len(matches)>0):
                for i in range(len(matches)):
                    if matches[i][1] > score :
                        matches.insert(i,(detector,score))
                        break
                else:
                    matches.append(i,(detector,score))
            else:
                matches.append(
                        (detector,score)
                        )
        print()
        return matches

if __name__ == "__main__" :
    import os
    from eff_word_net.streams import SimpleMicStream
    from eff_word_net import samples_loc
    print(samples_loc)


    base_model = Resnet50_Arc_loss()
    
    mycroft_hw = HotwordDetector(
        hotword="mycroft",
        model = base_model,
        reference_file=os.path.join(samples_loc,"mycroft_ref.json"),
        threshold=0.7,
        relaxation_time=2
    )
    
    alexa_hw = HotwordDetector(
            hotword="alexa",
            model=base_model,
            reference_file=os.path.join(samples_loc,"alexa_ref.json"),
            threshold=0.7,
            relaxation_time=2,
            #verbose=True
    )
    
    balloon_hw = HotwordDetector(
        hotword="balloon",
        model=base_model,
        reference_file=os.path.join(samples_loc,"balloon_ref.json"),
        threshold=0.65,
        relaxation_time=2,
        #verbose=True
    )
    
    computer_hw = HotwordDetector(
        hotword="computer",
        model=base_model,
        reference_file=os.path.join(samples_loc,"computer_ref.json"),
        threshold=0.7,
        relaxation_time=2,
        #verbose=True
    )
    
    mobile_hw = HotwordDetector(
        hotword="mobile",
        model = base_model,
        reference_file=os.path.join(samples_loc,"mobile_ref.json"),
        threshold=0.7,
        relaxation_time=2,
        #verbose=True
    )
    
    lights_on = HotwordDetector(
        hotword="lights_on",
        model = base_model,
        reference_file=os.path.join(samples_loc,"lights_on_ref.json"),
        threshold=0.7,
        relaxation_time=2    
    )
    
    
    lights_off = HotwordDetector(
        hotword="lights_on",
        model = base_model,
        reference_file=os.path.join(samples_loc,"lights_off_ref.json"),
        threshold=0.7,
        relaxation_time=2    
    )
    
    multi_hotword_detector = MultiHotwordDetector(
        [mycroft_hw, alexa_hw, balloon_hw, computer_hw, mobile_hw, lights_on, lights_off],
        model=base_model,
        continuous=True,
    )
    
    mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=0.75)
    mic_stream.start_stream()

    print("Say ", " / ".join([x.hotword for x in multi_hotword_detector.detector_collection]))

    while True :
        frame = mic_stream.getFrame()
        result = multi_hotword_detector.findBestMatch(frame)
        if(None not in result):
            print(result[0],f",Confidence {result[1]:0.4f}")
