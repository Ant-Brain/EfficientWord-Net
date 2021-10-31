"""
Can be run directly in cli 
`python -m efficientword.generate_reference`
"""
from eff_word_net.audio_processing import audioToVector, fixPaddingIssues
import os , glob
import numpy as np
import json
from eff_word_net.package_installation_scripts import check_install_librosa

check_install_librosa()

import librosa


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
                    fixPaddingIssues(x)
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
