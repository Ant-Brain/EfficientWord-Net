"""
Needs to be run directly in cli via
`python -m eff_word_net.ibm_generate`

Can be used to artificially synthesize audio samples for a given hotword
Uses ibm's demo of cloud neural voice , hence try to use it as low as possible
"""

import requests
import json
import shutil
from os.path import isdir, join
from os import mkdir
from time import sleep

def _getSoundFile(word:str,voice:str,out_dir:str):
    assert isdir(out_dir) , "Not a valid output directory"

    out_dir = join(out_dir,word.replace(" ","_"))
    session = requests.Session()

    data = {
        "ssmlText":f"<prosody pitch=\"default\" rate=\"-0%\">{word}</prosody>",
        "sessionID":"40fdfaa2-c2e1-46fa-baa5-efa85922eb3e"
        }

    response = session.post(
        "https://www.ibm.com/demos/live/tts-demo/api/tts/store",
        data=data
        )
    response_json = json.loads(response.text)
    if(response_json["status"]!="success"):
        print("first request failed")
        return False
    token = response_json["message"].split(" ")[-1]
    audio_response = session.get(
        f"https://www.ibm.com/demos/live/tts-demo/api/tts/newSynthesizer?id={token}&voice={voice}",stream = True)
    print(audio_response.status_code)
    if(audio_response.status_code==200):
        audio_response.raw.decode_content = True
        if(not isdir(out_dir)):
            mkdir(out_dir)
        with open(join(out_dir,f"{word}_{voice}.mp3"),'wb') as f:
            shutil.copyfileobj(audio_response.raw,f)
        return True
    return False

USA_VOICES = ["en-US_OliviaV3Voice","en-US_HenryV3Voice","en-US_MichaelV3Voice","en-US_AllisonV3Voice"]
UK_VOICES = ["en-GB_CharlotteV3Voice","en-GB_KateV3Voice","en-GB_JamesV3Voice"]

if __name__=="__main__":
    WORD = str(input("Enter your wakeword:"))
    PATH = str(input("Enter location where audio files will be saved:"))
    for voice in [*USA_VOICES,*UK_VOICES]:
        print(voice)
        if _getSoundFile(WORD,voice,PATH) :
            print("success")
        else:
            print("failed")
        sleep(2)
