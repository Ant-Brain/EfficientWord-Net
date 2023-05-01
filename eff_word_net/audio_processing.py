import glob
import tflite_runtime.interpreter as tflite
import os
import numpy as np
import random
from pprint import pprint
import json
import onnxruntime as rt

from eff_word_net.audio_utils import logfbank

LIB_FOLDER_LOCATION = os.path.dirname(os.path.realpath(__file__))

class ModelRawBackend :
    def __init__(self) :
        self.window_length = None
        self.window_frames = None
        pass 

    def _randomCrop(self, x:np.array,length=16000)->np.array :
        assert(x.shape[0]>self.window_frames)
        frontBits = random.randint(0,x.shape[0]-length) 
        return x[frontBits:frontBits+length]
    
    def _addPadding(self, x:np.array,length=16000)->np.array :
        assert(x.shape[0]<length)
        bitCountToBeAdded = length - x.shape[0]
        frontBits = random.randint(0,bitCountToBeAdded)
        #print(frontBits, bitCountToBeAdded-frontBits)
        new_x = np.append(np.zeros(frontBits),x)
        new_x = np.append(new_x,np.zeros(bitCountToBeAdded-frontBits))
        return new_x
    
    def _removeExistingPadding(self, x:np.array)->np.array:
        lastZeroBitBeforeAudio = 0 
        firstZeroBitAfterAudio = len(x)
        for i in range(len(x)):
          if x[i]==0:
            lastZeroBitBeforeAudio = i
          else:
            break
        for i in range(len(x)-1,1,-1):
          if x[i]==0:
            firstZeroBitAfterAudio = i
          else:
            break
        return x[lastZeroBitBeforeAudio:firstZeroBitAfterAudio]
    
    def fixPaddingIssues(self, x:np.array)-> np.array:
        x = self._removeExistingPadding(x)
        #print("Preprocessing Shape",x.shape[0])
        if(x.shape[0]>self.window_frames):
          return self._randomCrop(x,length=self.window_frames)
        elif(x.shape[0]<self.window_frames):
          return self._addPadding(x,length=self.window_frames)
        else:
          return x

    def scoreVector(self, inp_vector:np.array, embeddings:np.array) -> np.array:
        raise NotImplementedError("Vector scoring attempted on raw model backend")
    
    def audioToVector(self, inpAudio:np.array) -> np.array :
        raise NotImplementedError("Vector Convertion on raw model backend invoked")

class First_Iteration_Siamese(ModelRawBackend) :
    def __init__(self) :
        self.window_length = 1.0 # 1 second
        self.window_frames = int(self.window_length * 16000)     
        self.logmelcalc_interpreter = tflite.Interpreter(
                model_path=os.path.join(LIB_FOLDER_LOCATION,"models/first_iteration_siamese/logmelcalc.tflite"
            )
        )
        self.logmelcalc_interpreter.allocate_tensors()
        
        self.input_index = self.logmelcalc_interpreter.get_input_details()[0]["index"]
        self.output_details = self.logmelcalc_interpreter.get_output_details()
        
        self.baseModel_interpreter = tflite.Interpreter(
              model_path=os.path.join(LIB_FOLDER_LOCATION,"models/first_iteration_siamese/baseModel.tflite")
          )
        self.baseModel_interpreter.allocate_tensors()
        
        self.base_model_inp = self.baseModel_interpreter.get_input_details()
        self.base_model_out = self.baseModel_interpreter.get_output_details()


    def scoreVector(self, inp_vec, embeddings):
        """
        **Use this directly only if u know what you are doing**

        Returns a float with confidence of match 0 - 1
        """

        assert inp_vec.shape == (1,128), \
            "Inp vector should be of shape (1,128)"

        distances = np.sqrt(
            np.sum(
                (inp_vec - embeddings)**2,
                axis=1
            )
        )

        distances[distances>0.3] = 0.3
        top3 = (0.3-np.sort(distances)[:3])/0.3
        out = 0.0
        for i in top3 :
            out+= (1-out) * i

        return out


    def audioToVector(self, inpAudio:np.array) -> np.array :
        """
        Converts 16000Hz sampled 1 sec of audio to vector embedding
        Inp Parameters :
    
            inpAudio  : np.array of shape (16000,)
    
        Out Parameters :
    
            1 vector embedding of shape (128,1)
    
        """
        assert(inpAudio.shape==(self.window_frames,))
    
        self.logmelcalc_interpreter.set_tensor(
            self.input_index,
            np.expand_dims(
                inpAudio/inpAudio.max(),
                axis=0
            ).astype("float32")
        )
        self.logmelcalc_interpreter.invoke()
        self.logmel_output = self.logmelcalc_interpreter.get_tensor(self.output_details[0]['index'])
        self.baseModel_interpreter.set_tensor(
            self.base_model_inp[0]["index"],
            np.expand_dims(self.logmel_output,axis=(0,-1)).astype("float32")
        )
        self.baseModel_interpreter.invoke()
        output_data = self.baseModel_interpreter.get_tensor(self.base_model_out[0]['index'])
    
        return output_data
    
class Resnet50_Arc_loss(ModelRawBackend):
    def __init__(self):
        
        self.window_length = 1.5
        self.window_frames = int(self.window_length * 16000)

        self.onnx_sess = rt.InferenceSession(
            os.path.join(
            LIB_FOLDER_LOCATION, "models/resnet_50_arc/slim_93%_accuracy_72.7390%.onnx"),
            sess_options= rt.SessionOptions(),
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name:str = self.onnx_sess.get_inputs()[0].name
        self.output_name:str = self.onnx_sess.get_outputs()[0].name

        self.audioToVector(np.float32(np.zeros(self.window_frames,))) #warmup inference

    def compute_logfbank_features(self, inpAudio:np.array)->np.array:
        """
        This assumes a mono channel input
        """
        return logfbank(
           inpAudio,
           samplerate=16000,
           winlen=0.025,
           winstep=0.01,
           nfilt=64,
           nfft=512,
           preemph=0.0
        )
    
    def scoreVector(self, inp_vector: np.array, embeddings: np.array) -> np.array:
        #print(inp_vector.shape, embeddings.shape)
        #inp_norm = np.sqrt(np.sum(inp_vector**2, axis=1))
        #embeddings_norm = np.sqrt(np.sum(embeddings**2, axis=1))
        #print(inp_norm, embeddings_norm)

        #inp_vector = inp_vector/  inp_norm
        #embeddings = embeddings/ np.expand_dims(embeddings_norm, axis = -1)

        #inp_vector = inp_vector / np.linalg.norm(inp_vector)
        #embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=0), axis = -1)

        cosine_similarity = np.matmul(embeddings, inp_vector.T)
        #print(cosine_similarity)
        confidence_scores = (cosine_similarity+1)/2
        #print(confidence_scores.max())
        #print(cosine_similarity.shape, cosine_similarity.max())

        return confidence_scores.max()

    def audioToVector(self, inpAudio: np.array) -> np.array:
        
        assert inpAudio.shape == (self.window_frames, ) #1.5 sec long window
        features = self.compute_logfbank_features(inpAudio)
        #features_norm = self.min_max_normalize_features(features)

        output = self.onnx_sess.run(
           [self.output_name],
           {
                self.input_name : np.float32(
                    np.expand_dims(
                        features,
                        #features_norm,
                        axis = (0,1) # adding channel and batch dimension
                    )
                )
           }
        )[0]
        
        return output


from enum import Enum

class ModelType(str, Enum):
    first_iteration_siamese = "first_iteration_siamese"
    resnet_50_arc = "resnet_50_arc"

MODEL_TYPE_MAPPER = {
    "first_iteration_siamese" : First_Iteration_Siamese,
    "resnet_50_arc" : Resnet50_Arc_loss
}
