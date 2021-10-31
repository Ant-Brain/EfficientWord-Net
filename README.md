# EfficientWord-Net
![Versions : 3.6 ,3.7,3.8,3.9](https://camo.githubusercontent.com/a7b5b417de938c1faf3602c7f48f26fde8761a977be85390fd6c0d191e210ba8/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f74656e736f72666c6f772e7376673f7374796c653d706c6173746963)

## Hotword detection based on one-shot learning

EfficientWord-Net is an hotword detection engine based on one-shot
learning inspired from FaceNet's Siamese Network Architecture.

This repository is an official implemenation of EfficientWord-Net as
a python library from the authors.

The library is purely written with python and uses Google's Tflite
implemenation for faster realtime inference.

## Access preprint

The paper is currently under review, once published all the preprint and the training code will be available for public access
<br>

## Python Version Requirements

This Library works between python versions:
    3.6 to 3.9
<br>

## Dependencies Installation
Before running the pip installation command for the library, few dependencies need to be installed manually

* [PyAudio (depends on PortAudio)](https://abhgog.gitbooks.io/pyaudio-manual/content/installation.html)
* [Tflite (tensorflow lightweight binaries)](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python)
* [Librosa (Binaries might not be available for certain systems)](https://github.com/librosa/librosa)
Mac OS M* and Raspberry Pi users might have to compile these dependecies

***tflite*** package cannot be listed in requirements.txt hence will be automatically installed when the package is initialized in the system

***librosa*** package is not required for inference only cases , however when generate_reference is called , will be automatically installed

<br>

## Package Installation
Run the following pip command

```
pip install EfficientWord-Net
```

and to import running

```
import eff_word_net
```
<br>

## Demo
After installing the packages, you can run the Demo
script inbuilt with library (ensure you have a working mic)

Command to run demo
```
python -m eff_word_net.engine
```
<br>

## Generating Custom Wakewords

For any new hotword, the library needs information about the hotword, this
information is obtained from a file called `{wakeword}_ref.json` . 
Eg: For the wakeword 'alexa' , the library would need the file called `alexa_ref.json`

These files can be generated with the following procedure:

One needs to collect few 4 to 10 uniquely sounding pronunciations
of a given wakeword. Then put them into a seperate folder, which doesnt contain 
anything else.

Finally run this command, it will ask for the input folder's location 
(containing the audio files) and the output folder (where _ref.json file will be stored).
```
python -m eff_word_net.generate_reference
```

The pathname of the generated wakeword needs to passed to the HotwordDetector detector instance.

```python
HotwordDetector(
        hotword="hello",
        reference_file = "/full/path/name/of/hello_ref.json")
)
```

Few wakewords such as **Google**, **Firefox**, **Alexa**, **Mobile**, **Siri** the library has predefined embeddings readily available in the library installation directory, its path is readily available in the following variable

```python
from eff_word_net import samples_loc
```

<br>


## Try your first single hotword detection script

```python
import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net import samples_loc

alexa_hw = HotwordDetector(
        hotword="Alexa",
        reference_file = os.path.join(samples_loc,"alexa_ref.json"),
    )

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Alexa ")
while True :
    frame = mic_stream.getFrame()
    result = alexa_hw.checkFrame(frame)
    if(result):
        print("Wakeword uttered")

```
<br>


## Detecting Mulitple Hotwords from audio streams

The library provides a computation friendly way 
to detect multiple hotwords from a given stream, installed
of running `checkFrame()` of each wakeword individually

```python
import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net import samples_loc

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
    ) # Efficient multi hotword detector

mic_stream = SimpleMicStream()
mic_stream.start_stream()

print("Say Google / Alexa / Siri")
while True :
    frame = mic_stream.getFrame()
    result = multi_hw_engine.findBestMatch(frame)
    if(None not in result):
        print(result[0],f",Confidence {result[1]:0.4f}")
```
<br>

## TODO :
* Add contribute section in README
* Add audio file handler in streams
