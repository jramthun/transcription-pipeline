import pandas as pd
import numpy as np

def match_speaker_segments(contentId, fill_nearest=False):
    # Convert RTTM to dataframe so that we can match with relevant text
    pred_rttm = pd.read_csv(f"_msdd_output/pred_rttms/{contentId}.rttm", sep="\s+", header=None, names=["SPEAKER", "filename", "channel", "turnOnset", "turnDuration", "orthographyField", "speakerType", "speakerName", "confidenceScore", "signalLookahead"])
    pred_rttm["start"] = pred_rttm["turnOnset"]
    pred_rttm["end"] = pred_rttm["turnOnset"] + pred_rttm["turnDuration"]

    # Read aligned segments
    pred_segments = pd.read_json(f"_transcripts/segments/{contentId}.json")

    for idx, seg in pred_segments.iterrows():
        # assign speaker to segment (if any) by computing segment intersection
        pred_rttm['intersection'] = np.minimum(pred_rttm['end'], seg['end']) - np.maximum(pred_rttm['start'], seg['start']) # segment overlap duration
        
        # Otherwise we look for closest speaker-segment match
        if not fill_nearest:
            tmp_df = pred_rttm[pred_rttm['intersection'] > 0]
        else:
            tmp_df = pred_rttm

        if len(tmp_df) > 0:
            # sum over speakers to select the best match
            speaker = tmp_df.groupby("speakerName")["intersection"].sum().sort_values(ascending=False).index[0]
            pred_segments.at[idx, "speaker"] = speaker

    # Save updated segments
    pred_segments.to_json(f"{contentId}-out.json")

    # Build labelled transcript as list
    transcript_results = []
    previous_speaker = ''

    for _, segment in pred_segments.iterrows():
        speaker = segment["speaker"]
        text = segment["text"].strip()

        # If this speaker doesn't match the previous one, start a new group
        if speaker != previous_speaker:
            transcript_results.append({"speaker": speaker, "text": text})
            previous_speaker = speaker
        else:
            transcript_results[-1]["text"] += f" {text}"

    # Write transcript to file with speaker labels
    with open(f"pred_transcript-{contentId}.txt", "w") as f:
        for segment in transcript_results:
            f.write(f"{segment['speaker']}: {segment['text'].strip()}\n")

    return transcript_results

if __name__ == "__main__":
    match_speaker_segments("finance120524")