import glob
import librosa
import tflite_runtime.interpreter as tflite
import os
import numpy as np
import random
from pprint import pprint
import json

location = os.path.dirname(os.path.realpath(__file__))

logmelcalc_interpreter = tflite.Interpreter(
        model_path=os.path.join(location,"logmelcalc.tflite"
    )
)

logmelcalc_interpreter.allocate_tensors()

input_index = logmelcalc_interpreter.get_input_details()[0]["index"]
output_details = logmelcalc_interpreter.get_output_details()

baseModel_interpreter = tflite.Interpreter(
        model_path=os.path.join(location,"./baseModel.tflite")
    )
baseModel_interpreter.allocate_tensors()

base_model_inp = baseModel_interpreter.get_input_details()
base_model_out = baseModel_interpreter.get_output_details()

def _randomCrop(x:np.array,length=16000)->np.array :
    assert(x.shape[0]>length)
    frontBits = random.randint(0,x.shape[0]-length) 
    return x[frontBits:frontBits+length]

def _addPadding(x:np.array,length=16000)->np.array :
    assert(x.shape[0]<length)
    bitCountToBeAdded = length - x.shape[0]
    frontBits = random.randint(0,bitCountToBeAdded)
    #print(frontBits, bitCountToBeAdded-frontBits)
    new_x = np.append(np.zeros(frontBits),x)
    new_x = np.append(new_x,np.zeros(bitCountToBeAdded-frontBits))
    return new_x

def _removeExistingPadding(x:np.array)->np.array:
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

def _fixPaddingIssues(x:np.array,length=16000)-> np.array:
    x = _removeExistingPadding(x)
    #print("Preprocessing Shape",x.shape[0])
    if(x.shape[0]>16000):
      return _randomCrop(x,length=length)
    elif(x.shape[0]<16000):
      return _addPadding(x,length=length)
    else:
      return x

def audioToVector(inpAudio:np.array) -> np.array :
    """
    Converts 16000Hz sampled 1 sec of audio to vector embedding
    Inp Parameters :
        inpAudio  : np.array of shape (16000,)
    Out Parameters :
        1 vector embedding of shape (128,1)
    """
    assert(inpAudio.shape==(16000,))

    logmelcalc_interpreter.set_tensor(input_index,np.expand_dims(inpAudio/inpAudio.max(),axis=0).astype("float32"))
    logmelcalc_interpreter.invoke()
    logmel_output = logmelcalc_interpreter.get_tensor(output_details[0]['index'])
    baseModel_interpreter.set_tensor(
        base_model_inp[0]["index"],
        np.expand_dims(logmel_output,axis=(0,-1)).astype("float32")
    )
    baseModel_interpreter.invoke()
    output_data = baseModel_interpreter.get_tensor(base_model_out[0]['index'])

    return output_data


def generate_reference_file(input_dir:str,output_dir:str,wakeword:str,debug:bool=False):
    """
    Generates reference files for few shot learning comparison

    Inp Parameters:

        input_dir : directory which holds only sample audio files
        of wakeword

        output_dir: directory where generated reference file will
        be stored

        debug=False : when true prints out the distance matrix of
        the samples

    Out Parameters:
        None

    """

    assert(os.path.isdir(input_dir))
    assert(os.path.isdir(output_dir))
    embeddings = []

    audio_files = [
        *glob.glob(input_dir+"/*.mp3"),
        *glob.glob(input_dir+"/*.wav")
    ]

    for audio_file in audio_files :
        x,_ = librosa.load(audio_file,sr=16000)
        embeddings.append(
                audioToVector(
                    _fixPaddingIssues(x)
                )
            )

    embeddings = np.squeeze(np.array(embeddings))

    if(debug):
        distanceMatrix = []

        for embedding in embeddings :
            distanceMatrix.append(
                np.sqrt(np.sum((embedding-embeddings)**2,axis=1))
            )

        temp = np.squeeze(distanceMatrix).astype(np.float16)
        temp2 = temp.flatten()
        print(np.std(temp2),np.mean(temp2))
        print(temp)

    open(os.path.join(output_dir,f"{wakeword}_ref.json") ,'w').write(
            json.dumps(
                {
                    "embeddings":embeddings.astype(float).tolist()
                    }
                )
            )
if __name__ == "__main__" :
    generate_reference_file(
            input("Paste Path of folder Containing audio files:"),
            input("Paste Path of location to save *_ref.json :"),
            input("Enter Wakeword Name                                 :")
            )
