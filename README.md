
# EfficientWord
![Versions : 3.6 ,3.7,3.8,3.9](https://camo.githubusercontent.com/a7b5b417de938c1faf3602c7f48f26fde8761a977be85390fd6c0d191e210ba8/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f74656e736f72666c6f772e7376673f7374796c653d706c6173746963)
### Hotword detection based on one-shot learning

EfficientWord is an hotword detection engine based on one-shot
learning inspired from FaceNet's Siamese Network Architecture.

This repository is an official implemenation of EfficientNet as
a python library from the authors.

The library is purely written with python and uses Google's Tflite
implemenation for faster realtime inference.

### Access preprint

For a more detailed perspective of the project, please refer to
the [preprint]() of EfficientWord.

### Python Version Requirements

This Library works between python versions:
    3.6 to 

### Dependencies Installation

Before running the pip installation command for the library, few 
dependencies need to be installed manually
* [PyAudio (depends on PortAudio)](https://abhgog.gitbooks.io/pyaudio-manual/content/installation.html)
* [Tflite (tensorflow lightweight binaries)](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python)
* [Librosa (Binaries might not be available for certain systems)](https://github.com/librosa/librosa)
Mac OS M* users might have to compile these dependecies

### Package Installation
Run the following pip command

```
pip install efficientword
```

and to import running

```
import efficientword
```

### Demo
After installing the packages, you can run the Demo
script inbuilt with library (ensure you have a working mic)

Command to run demo
```
python -m efficientword.engine
```

### Try your first single hotword detection script

```
import os
from efficientword.streams import SimpleMicStream
from efficientword.engine import HotwordDetector
from efficientword import samples_loc

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

### Detecting Mulitple Hotwords from audio streams

The library provides a computation friendly way 
to detect multiple hotwords from a given stream, installed
of running `checkFrame()` of each wakeword individually

```
import os
from efficientword.streams import SimpleMicStream
from efficientword import samples_loc

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

### Generating Custom Wakewords

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
python -m efficientword.generate_reference
```

The pathname of the generated wakeword needs to passed to the HotwordDetector detector instance.

```
HotwordDetector(
        hotword="hello",
        reference_file = "/full/path/name/of/hello_ref.json")
)
```

### Contributing

* **Contributions through code :** We require some help in generalizing the logmelcalc tflite implementation, we invite tensorflow gurus out there to help us out (Current Implementation due to a bug can process only one 1sec audio frame at a time)

* **Contrubtions through suggestions :** Please refer though our [preprint]() and suggest architectural changes for better performance, we'd love to implement them in future releases.