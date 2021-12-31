import json
from os.path import isfile , join
import numpy as np
import pyaudio

from typing import Tuple , List

from eff_word_net.audio_processing import audioToVector
from eff_word_net import RATE

class HotwordDetector :

    """
    EfficientWord based HotwordDetector Engine implementation class
    """

    def __init__(
            self,
            hotword:str,
            reference_file:str,
            threshold:float=0.9,
            activation_count=2,
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

        self.__repeat_count = 0
        self.__activation_count = activation_count
        self.verbose = verbose

        self.__relaxation_time_step = 4 #number of cycles to prevent recall after a trigger
        self.__is_it_a_trigger = False

    def __repr__(self):
        return f"Hotword: {self.hotword}"

    def is_it_a_trigger(self):
        return self.__is_it_a_trigger

    def getMatchScoreVector(self,inp_vec:np.array) -> float :
        """
        **Use this directly only if u know what you are doing**

        Returns the match score from 0 to 1 for an embedding with
        given reference file
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

        #assert self.redundancy_count>0 , "redundancy_count count can only be greater than 0"

        self.__is_it_a_trigger = False

        if self.__repeat_count < 0 :
            self.__repeat_count += 1

        elif out > self.threshold :
            if self.__repeat_count == self.__activation_count -1 :
                self.__repeat_count = - self.__relaxation_time_step
                self.__is_it_a_trigger = True
            else:
                self.__repeat_count +=1

        elif self.__repeat_count > 0:
            self.__repeat_count -= 1

        return out

    def checkVector(self,inp_vec:np.array) -> bool:
        """
        **Use this directly only if u know what you are doing**

        Checks if given a given embedding matches with
        given reference file
        """

        assert inp_vec.shape == (1,128), \
            "Inp vector should be of shape (1,128)"

        score = self.getMatchScoreVector(inp_vec)

        return self.is_it_a_trigger() if self.continuous else score >= self.threshold

    def get_repeat_count(self)-> int :
        return self.__repeat_count

    def getMatchScoreFrame(
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
            processing continuous speech , to minimalize false positives

        **Note : change unsafe to True only if you know what you are doing**

        Out Parameters:

            float , ranges btw 0 to 1 . Higher value denoting higher match

        """

        """
        if(not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:RATE//10]
            )
            if(upperPoint > 0.2):
                return False
        """

        assert inp_audio_frame.shape == (RATE,), \
            f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        return self.getMatchScoreVector(
            audioToVector(
                inp_audio_frame
            )
            )


    def checkFrame(self,inp_audio_frame:np.array,unsafe:bool = False) -> bool :
        """
        Converts given audio frame to embedding and checks for similarity
        with given reference file

        Inp Parameters:

            inp_audio_frame : np.array of 1channel 1 sec 16000Hz sampled audio 
            frame
            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech , to minimalize false positives

        **Note : change unsafe to True only if you know what you are doing**

        Out Parameters:

            bool , conveys if given frame has a likely match of the hotword

        """

        assert inp_audio_frame.shape == (RATE,), \
            f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        """
        if(not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:RATE//10]
            )
            if(upperPoint > 0.2):
                return False
        """
        score = self.getMatchScoreFrame(inp_audio_frame)

        return self.is_it_a_trigger() if self.continuous else score >= self.threshold

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
            score = detector.getMatchScoreVector(embedding)
            if(self.continous):
                if(not detector.is_it_a_trigger()):
                    continue
            else:
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


        if(not unsafe):
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
            if(score<detector.threshold):
                continue
            if(score>best_match_score):
                best_match_score = score
                matches.append(
                        (detector,best_match_score)
                        )

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
            activation_count=3
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
