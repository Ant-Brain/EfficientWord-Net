"""
Can be run directly in cli 
`python -m eff_word_net.generate_reference`
"""
import os , glob
import numpy as np
import json
from eff_word_net.package_installation_scripts import check_install_librosa
from eff_word_net.audio_processing import (
    ModelType,
    MODEL_TYPE_MAPPER
)

check_install_librosa()

import librosa
import typer
from rich.progress import track

def generate_reference_file(
        input_dir:str = typer.Option(...),
        output_dir:str = typer.Option(...),
        wakeword:str = typer.Option(...),
        model_type:ModelType = typer.Option(..., case_sensitive=False),
        debug:bool=typer.Option(False)
    ):

    """
    Generates reference files for few shot learning comparison

    Inp Parameters:

        input_dir : directory which holds only sample audio files
        of wakeword

        output_dir: directory where generated reference file will
        be stored

        wakeword: name of the wakeword
        
        model_type: type of the model to be used
        
        debug: self explanatory
    Out Parameters:

        None

    """
    #print(model_type)
    model = MODEL_TYPE_MAPPER[model_type.value]()

    assert(os.path.isdir(input_dir))
    assert(os.path.isdir(output_dir))
    embeddings = []

    audio_files = [
        *glob.glob(input_dir+"/*.mp3"),
        *glob.glob(input_dir+"/*.wav")
    ]


    for audio_file in track(audio_files, description="Generating Embeddings.. ") :
        x,_ = librosa.load(audio_file,sr=16000)
        embeddings.append(
                model.audioToVector(
                    model.fixPaddingIssues(x)
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
                    "embeddings":embeddings.astype(float).tolist(),
                    "model_type":model_type.value
                    }
                )
            )

if __name__ == "__main__" :
    typer.run(generate_reference_file)