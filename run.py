# Package Imports
import argparse
import logging
import os

# Local imports
from utils.audio_utils import prepareAudio
from utils.transcribe import generateTranscript
from utils.diarize import createManifest, diarize
from utils.speaker_utils import match_speaker_segments

# Test/Debug Imports
import time


# Create and configure logger
if not os.path.exists('./_logs'):
    os.mkdir(path='./_logs')
logging.basicConfig(filename="_logs/newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Argument Parsing
parser = argparse.ArgumentParser(
    description='End-to-end solution for transcribing long-form video content. Designed with extended Congressional hearings and consumer hardware in mind.'
)

# Video/Stream URL
parser.add_argument('url', type=str,
    help='The URL to the stream manifest of the video to transcribe. See https://github.com/yt-dlp/yt-dlp for more information')

# Feature Control
parser.add_argument('--input_manifest', default=None, type=str,
    help='Path/to/input_manifest.json. The expected format is {unique_name: video_url}') # TODO: implement input manifest to allow for batch processing

parser.add_argument('--no-grammar', action='store_true',
                    help='Disable automatic correction of text segments. Provides minor speed up.')

parser.add_argument('--no-sep', action='store_true',
                    help='Controls whether to use diarization to separate speakers in the transcript') # i.e., setting this flag stops after generating the full transcript

args = parser.parse_args()

if args.input_manifest is not None:
    print("Using input manifest")

# Auto-correct to require diaraization for speaker-labeling
if args.label_speakers and not args.diar:
    args.diar = True

def runPipeline(URL):
#### Process Breakdown
## 1. Get video from the internet using stream manifest
    # Input: link to stream manifest
    # Output: video file
## 2. Extract audio from the video (as 16kHz WAV)
    # Input: video
    # Output: audio
    if not URL:
        URL = args.url

    content_id = prepareAudio(URL)
    # TODO: replace checkpoints with proper logging to clean up terminal
    # TODO: given flags, we could use tqdm to report status of the pipeline with a progress bar?
    print("--- PrepareAudio checkpoint: %s seconds ---" % (time.time() - start_time))

## 3. Transcribe the audio
    # Input: audio
    # Output: transcript
    generateTranscript(content_id)
    print('--- GenerateTranscript checkpoint: %s seconds ---' % (time.time() - start_time))

## --- First stopping point: completed transcript of the video


## 4. Forced Alignment
    # Currently handled as part of WhisperX, may need to recreate if using another model
## 5. Diarization - https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
    # Note: NVIDIA has not yet release MSDD_meeting or MSDD_general yet, so we must use MSDD_telephonic and a custom yml config
    # Input: input manifest (audio, transcript, etc)
    # Output: output manifests, predicted speaker/speech clusters + interim files from VAD and diarization
    createManifest(content_id)
    print('--- CreateManifest checkpoint: %s seconds ---' % (time.time() - start_time))
    diarize(content_id)
    print('--- Diarize checkpoint: %s seconds ---' % (time.time() - start_time))

## 6. Combining the outputs
    # Input: aligned speech, diarized speech, transcript
    # Output: combined transcript with speaker labels
    match_speaker_segments(content_id)
    print('--- MatchSpeakerSegments checkpoint: %s seconds ---' % (time.time() - start_time))

## --- Second stopping point: completed transcript of the video, with speaker separation represented


#### Future directions -> TODO: move this to a separate file
## A. Using NLP to determine who each speaker is using the transcript (so we don't have to train a model on 538 people's speech every two years)
    # Output: final transcript with labeled speakers, nicely formatted in a generic style
## B. Converting the transcript to a RAG-compatible format to allow for summarization, DocQA, and other NLP tasks
    # Output: RAG-compatible (chroma/vector) database
    # Further: Allowing for a user (in a ChatGPT-like interface) to ask questions about the video content using the transcript
        # Afaik, there are no available multi-modal models capable of doing this yet, at least not with 10+ hours of video
## C. Constructing subtitles (including speaker labels) from the transcript to allow for evaluation, RLHF tuning, and/or manual correction by the user
    # Output: subtitle file, option to embed directly into the video (if necessary)


if __name__ == "__main__":
    start_time = time.time()
    runPipeline()
    print("--- Total Runtime: %s seconds ---" % (time.time() - start_time))