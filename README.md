# EfficientWord-Net: Hotword Detection Based on Few-Shot Learning

Home assistants require special phrases called hotwords to get activated (e.g., "OK Google"). EfficientWord-Net is a hotword detection engine based on few-shot learning that allows developers to add custom hotwords to their programs without extra charges. The library is purely written in Python and uses Google's TFLite implementation for faster real-time inference. It is inspired by FaceNet's Siamese Network Architecture and performs best when 3-4 hotword samples are collected directly from the user.

### Demo of EfficientWord-Net on Pi

<https://user-images.githubusercontent.com/44740048/139785995-3330d65a-cfe1-4e92-8769-ee389a122acc.mp4>

## Access Training File

[Training File](./training.ipynb) to access the training file.

## Datasets

Checkout the people speech spoken words dataset, the Resnet_50_Arc_loss was trained on the same  <https://mlcommons.org/datasets/multilingual-spoken-words/>

## Access Paper

[Research Paper](https://worldscientific.com/doi/10.1142/S0219649222500599) to access the research paper.

## Python Version Requirements

This library works with Python versions 3.10 to 3.14.

## Dependencies Installation

Before running the pip installation command for the library, a few dependencies need to be installed manually:

- [PyAudio (depends on PortAudio)](https://abhgog.gitbooks.io/pyaudio-manual/content/installation.html)

## Package Installation

Run the following pip command:

```
pip install EfficientWord-Net
```

To import the package:

```python
import eff_word_net
```

## Demo

After installing the packages, you can run the demo script built into the library (ensure you have a working microphone).

Access documentation from: <https://ant-brain.github.io/EfficientWord-Net/>

Command to run the demo:

```
python -m eff_word_net.engine
```

## Generating Custom Wakewords

For any new hotword, the library needs information about the hotword. This information is obtained from a file called `{wakeword}_ref.json`.
For example, for the wakeword 'alexa', the library would need the file called `alexa_ref.json`.

These files can be generated with the following procedure:

1. Collect 4 to 10 uniquely sounding pronunciations of a given wakeword. Put them into a separate folder that doesn't contain anything else.

2. Finally, run this command. It will ask for the input folder's location (containing the audio files) and the output folder (where the _ref.json file will be stored):

```
python -m eff_word_net.generate_reference
```

Note: Only WAV files are supported. If your audio files are in other formats, convert them to WAV using https://ffmpegwasm.netlify.app/playground, which performs conversion entirely in your browser without uploading files.

Once you have generated the reference file, you can test the hotword detection using:

```
python -m eff_word_net.single_word_test --reference-file /full/path/to/your_wakeword_ref.json --hotword "your_wakeword"
```

Ensure you have a working microphone.

The pathname of the generated wakeword needs to be passed to the HotwordDetector instance:

```python
HotwordDetector(
    hotword="hello",
    model=Resnet_50_Arc_loss(),
    reference_file="/full/path/name/of/hello_ref.json",
    threshold=0.9,  # min confidence required to consider a trigger
    relaxation_time=0.8  # default value, in seconds
)
```

The model variable can receive an instance of Resnet_50_Arc_loss or First_Iteration_Siamese.

The relaxation_time parameter is used to determine the minimum time between any two triggers. Any potential triggers before the relaxation_time will be canceled. The detector operates on a sliding window approach, resulting in multiple triggers for a single utterance of a hotword. The relaxation_time parameter can be used to control multiple triggers; in most cases, 0.8 seconds (default) will suffice.

## Out-of-the-Box Sample Hotwords

The library has predefined embeddings readily available for a few wakewords such as **Mycroft**, **Google**, **Firefox**, **Alexa**, **Mobile**, and **Siri**. Their paths are readily available in the library installation directory.

```python
from eff_word_net import samples_loc
```

<br>

## Try your first single hotword detection script

```python
import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector

from eff_word_net.audio_processing import Resnet50_Arc_loss

from eff_word_net import samples_loc

base_model = Resnet50_Arc_loss()

mycroft_hw = HotwordDetector(
    hotword="mycroft",
    model = base_model,
    reference_file=os.path.join(samples_loc, "mycroft_ref.json"),
    threshold=0.7,
    relaxation_time=2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)

mic_stream.start_stream()

print("Say Mycroft ")
while True :
    frame = mic_stream.getFrame()
    result = mycroft_hw.scoreFrame(frame)
    if result==None :
        #no voice activity
        continue
    if(result["match"]):
        print("Wakeword uttered",result["confidence"])

```

<br>

## Detecting Mulitple Hotwords from audio streams

The library provides a computation friendly way
to detect multiple hotwords from a given stream, instead of running `scoreFrame()` of each wakeword individually

```python

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


computer_hw = HotwordDetector(
    hotword="computer",
    model=base_model,
    reference_file=os.path.join(samples_loc,"computer_ref.json"),
    threshold=0.7,
    relaxation_time=2,
    #verbose=True
)

multi_hotword_detector = MultiHotwordDetector(
    [mycroft_hw, alexa_hw, computer_hw],
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


```

<br>

Access documentation of the library from here : <https://ant-brain.github.io/EfficientWord-Net/>

Here's the corrected version of the README.md file with improved grammar and formatting:

## Want the same in C++ have a look at the following gist

@Stypox has provided a gist with which one can perform inference for the same in C++ <https://gist.github.com/Stypox/2bf47017ea5361776d9c096e886e0540>

## Change notes from v1.0.3 to v1.0.4

- Stability related improvements

## Change notes from v1.0.4 to v1.0.5

- Pivoted from pydub to soundfile for audio file processing to improve reliability.
- Dropped support for MP3 files in reference generation.
- Updated documentation to guide users on converting non-WAV files.
- Added single_word_test module with typer support for testing custom reference files.

## Change notes from v1.0.2 to v1.0.3

- Pivoted to more cross platform friendly dependencies
- Depreceated tensorflow model due to inference difficulties of tflite in platforms like Windows

## Change notes from 0.2.2 to v1.0.1

### New Model Addition: Resnet_50_Arc_loss with huge improvements

- Trained a new model from scratch using a modified distilled dataset from MLCommons.
- Used Arc loss function instead of triplet loss function.
- The resultant model is stored as resnet_50_arcloss.
- The newer model showcases much better resilience towards background noise and requires fewer samples for good accuracy.
- Minor changes in the API flow to facilitate easy addition of newer models.
- Newer model can handle a fixed window length of 1.5 seconds.
- The old model can still be accessed through first_iteration_siamese.

## Change notes from v0.1.1 to 0.2.2

- Major changes to replace complex logic of handling poly triggers per utterance with simpler logic and a more straightforward API for programmers.
- Introduces breaking changes.
- C++ implementation of the current model is [here](https://github.com/Ant-Brain/EfficientWord-Net/issues/56).

## Limitations in Current Model

- Trained on single words, hence may result in bizarre behavior when using phrases like "Hey xxx".
- Audio processing window limited to 1 sec. Hence, it will not work effectively for longer hotwords.

## FAQ

- **Hotword Performance is bad**: If you are experiencing issues like this, feel free to ask in the [discussions](https://github.com/Ant-Brain/EfficientWord-Net/discussions/4).
- **Can it run on FPGAs like Arduino?**: No, the new Resnet_50_Arcloss model is too heavy to run on Arduino (roughly 88MB in size). We will soon add support for pruned versions of the model so that it can become light enough to run on tiny devices. For now, it should be able to run on Raspberry Pi-like devices.

## Contribution

- If you have ideas to make the project better, feel free to ping us in the [discussions](https://github.com/Ant-Brain/EfficientWord-Net/discussions/3).

## TODO

- Add audio file handler in streams. PRs are welcome.
- Add more detailed documentation explaining the sliding window concept.
- Add model fine-tuning support.
- Add support for sparse and fine-grained pruning where the resultant models could be used for fine-tuning (already working on this).

## Support Us

Our hotword detector's performance is notably lower compared to Porcupine. We have thought about better NN architectures for the engine and hope to outperform Porcupine. This has been our undergrad project, so your support and encouragement will motivate us to develop the engine further. If you love this project, recommend it to your peers, give us a 🌟 on GitHub, and a clap 👏 on [Medium](https://link.medium.com/yMBmWGM03kb).

Update: Your stars encouraged us to create a new model which is far better. Let's make this community grow!

## License

[Apache License 2.0](/LICENSE.md)
