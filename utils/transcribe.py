import whisperx
import gc 
import torch
import json
import os
from tqdm import tqdm
from text_utils import correct_text

device = "cuda" 
audio_file = "audio.mp3"
batch_size = 32 # reduce if low on GPU mem
compute_type = "bfloat16" # change to "int8" if low on GPU mem (may reduce accuracy)

def generateTranscript(audio_file):
    """
    Generate transcript for a given audio file using WhisperX.

    Parameters
    ----------
    audio_file : str
        The name of the audio file to transcribe.

    Returns
    -------
    transcript : dict
        A dictionary containing the transcribed text and its segments.

    Notes
    -----
    This function first transcribes the given audio file using the WhisperX model.
    It then aligns the transcript segments using the WhisperX align model.
    Finally, it saves the transcribed text and its segments to disk.
    """
    # 1. Transcribe with original whisper (batched)
    # save model to local path (optional)

    model_dir = "./_models"
    if not os.path.exists(model_dir):
        os.mkdir(path=model_dir)

    model = whisperx.load_model("distil-large-v3", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(f"_audio/{audio_file}.wav")
    result = model.transcribe(audio, batch_size=batch_size)

    gc.collect()
    torch.cuda.empty_cache()
    del model

    # 2. Align whisper output
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(transcript=result["segments"], model=align_model, align_model_metadata=metadata, audio=audio, device=device, return_char_alignments=False)
    
    gc.collect()
    torch.cuda.empty_cache()
    del align_model

    if not os.path.exists('./_transcripts'):
        os.mkdir(path='./_transcripts')
    if not os.path.exists('./_transcripts/text'):
        os.mkdir(path='./_transcripts/text')
    if not os.path.exists('./_transcripts/segments'):
        os.mkdir(path='./_transcripts/segments')

    # 3. Correct text segments via Grammarly editor model
    # TODO: implement flag to enable/disable correction
    correct_text(result["segments"])

    result["text"] = " ".join([seg["text"].strip() for seg in result["segments"]])

    # 4. Save results
    with open(f"_transcripts/text/{audio_file}.txt", "w") as f:
        f.write(result["text"])
    
    with open(f"_transcripts/segments/{audio_file}.json", "w") as f:
        f.write(json.dumps(result["segments"]))

    return result["segments"]

if __name__ == "__main__":
    generateTranscript(audio_file)