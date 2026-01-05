import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
import typer


def single_word_test(
    reference_file: str = typer.Option(
        ..., help="Path to the reference file (.json) for the hotword"
    ),
    hotword: str = typer.Option(..., help="Name of the hotword"),
    threshold: float = typer.Option(0.7, help="Detection threshold"),
    relaxation_time: float = typer.Option(2.0, help="Relaxation time in seconds"),
    window_length_secs: float = typer.Option(1.5, help="Window length in seconds"),
    sliding_window_secs: float = typer.Option(0.75, help="Sliding window in seconds"),
):
    base_model = Resnet50_Arc_loss()

    hw_detector = HotwordDetector(
        hotword=hotword,
        model=base_model,
        reference_file=reference_file,
        threshold=threshold,
        relaxation_time=relaxation_time,
    )

    mic_stream = SimpleMicStream(
        window_length_secs=window_length_secs,
        sliding_window_secs=sliding_window_secs,
    )

    mic_stream.start_stream()

    print(f"Say {hotword} ")
    while True:
        frame = mic_stream.getFrame()
        result = hw_detector.scoreFrame(frame)
        if result is None:
            continue
        if result["match"]:
            print("Wakeword uttered", result["confidence"])


if __name__ == "__main__":
    typer.run(single_word_test)
