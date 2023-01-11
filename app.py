import torch
import whisper
import os
import base64
from io import BytesIO
import whisper
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import format_timestamp
import numpy as np
import wget


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    checkpoint_path = "weights/large-v1.pt"
    with open(checkpoint_path, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cuda")



# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    print(model_inputs)
    global model
    # download the file
    wget.download(model_inputs["audio"], "input.mp3")
    model_inputs["audio"] = "input.mp3"
    # Run the model
    result = predict(**model_inputs)
    os.remove("input.mp3")
    # Return the results as a dictionary
    return result



def Input(
        default=None,
        choices=None,
        description=None,
    ):
    return default


def predict(
        audio: str = Input(
            description="path to the audio file",
        ),
        transcription: str = Input(
            choices=["plain text", "srt", "vtt"],
            default="plain text",
            description="Choose the format for the transcription",
        ),
        translate: bool = Input(
            default=False,
            description="Translate the text to English when set to True",
        ),
        language: str = Input(
            choices=sorted(LANGUAGES.keys())
            + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
        ),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        patience: float = Input(
            default=None,
            description="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        condition_on_previous_text: bool = Input(
            default=True,
            description="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        ),
        temperature_increment_on_fallback: float = Input(
            default=0.2,
            description="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        ),
        compression_ratio_threshold: float = Input(
            default=2.4,
            description="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        ),
        logprob_threshold: float = Input(
            default=-1.0,
            description="if the average log probability is lower than this value, treat the decoding as failed",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        )) -> dict:
    """
    Transcribe an audio file to text.
    """
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]

    args = {
        "language": language,
        "patience": patience,
        "suppress_tokens": suppress_tokens,
        "initial_prompt": initial_prompt,
        "condition_on_previous_text": condition_on_previous_text,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
    }

    result = model.transcribe(str(audio), temperature=temperature, **args)

    if transcription == "plain text":
        transcription = result["text"]
    elif transcription == "srt":
        transcription = write_srt(result["segments"])
    else:
        transcription = write_vtt(result["segments"])

    if translate:
        translation = model.transcribe(
            str(audio), task="translate", temperature=temperature, **args
        )

    return dict(
        segments=result["segments"],
        detected_language=LANGUAGES[result["language"]],
        transcription=transcription,
        translation=translation["text"] if translate else None,
    )


def write_vtt(transcript):
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result