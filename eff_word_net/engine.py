import json
from os.path import isfile , join
import numpy as np
import pyaudio

from typing import Tuple , List

from eff_word_net.audio_processing import audioToVector
from eff_word_net import RATE
from time import time as current_time_in_sec

class HotwordDetector :

    """
    EfficientWord based HotwordDetector Engine implementation class
    """

    def __init__(
            self,
            hotword:str,
            reference_file:str,
            threshold:float=0.9,
            relaxation_time=0.8,
            continuous=True,
            verbose = False):
        """
        Intializes hotword detector instance

        Inp Parameters:

            hotword : hotword in a string

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

        assert self.embeddings.shape[0]>4, \
            "Minimum of 4 sample datapoints is required"

        self.hotword = hotword
        self.threshold = threshold
        self.continuous = continuous

        self.relaxation_time = relaxation_time
        self.verbose = verbose

        self.__last_activation_time = current_time_in_sec()

    def __repr__(self):
        return f"Hotword: {self.hotword}"

    def __crossedRelaxationTime(self):
        return current_time_in_sec()-self.__last_activation_time > self.relaxation_time

    def scoreVector(self,inp_vec:np.array) -> float :
        """
        **Use this directly only if u know what you are doing**

        Returns a float with confidence of match 0 - 1
        """

        assert inp_vec.shape == (1,128), \
            "Inp vector should be of shape (1,128)"

        distances = np.sqrt(
            np.sum(
                (inp_vec - self.embeddings)**2,
                axis=1
            )
        )

        distances[distances>0.3] = 0.3
        top3 = (0.3-np.sort(distances)[:3])/0.3
        out = 0.0
        for i in top3 :
            out+= (1-out) * i

        if self.continuous :
            if not self.__crossedRelaxationTime() :
                return 0.001
            elif out>self.threshold :
                self.__last_activation_time = current_time_in_sec()

        return out


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

        assert inp_audio_frame.shape == (RATE,), \
            f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        score = self.scoreVector(
            audioToVector(
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
        assert inp_audio_frame.shape == (RATE,), \
            f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

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
        embedding = audioToVector(inp_audio_frame)

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
        assert inp_audio_frame.shape == (RATE,), \
            f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"


        if self.continous and (not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:1600]
            )
            if(upperPoint > 0.2 or upperPoint==0):
                return None , None

        embedding = audioToVector(inp_audio_frame)

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

    alexa_hw = HotwordDetector(
            hotword="Alexa",
            reference_file = os.path.join(samples_loc,"alexa_ref.json"),
        )

    siri_hw = HotwordDetector(
            hotword="Siri",
            reference_file = os.path.join(samples_loc,"siri_ref.json"),
        )

    mycroft_hw = HotwordDetector(
            hotword="mycroft",
            reference_file = os.path.join(samples_loc,"mycroft_ref.json"),
        )

    multi_hw_engine = MultiHotwordDetector(
            detector_collection = [
                alexa_hw,
                siri_hw,
                mycroft_hw,
            ],
        )

    mic_stream = SimpleMicStream()
    mic_stream.start_stream()

    print("Say Mycroft / Alexa / Siri")

    while True :
        frame = mic_stream.getFrame()
        result = multi_hw_engine.findBestMatch(frame)
        if(None not in result):
            print(result[0],f",Confidence {result[1]:0.4f}")
