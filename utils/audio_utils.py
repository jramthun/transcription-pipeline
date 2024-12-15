import json
import os
import yt_dlp
import re

# Example Hearing: Open Executive Session to Consider Favorably Reporting the Hearing the Nomination of David Samuel Johnson, of Virginia, to be Inspector General for Tax Administration, Department of the Treasury
# https://www.finance.senate.gov/hearings/open-executive-session-to-consider-favorably-reporting-the-hearing-the-nomination-of-david-samuel-johnson-of-virginia-to-be-inspector-general-for-tax-administration-department-of-the-treasury
URL = 'https://www-senate-gov-media-srs.akamaized.net/hls/live/2036795/finance/finance120524/master.m3u8'

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', }],
    'postprocessor_args': {'ffmpeg': ['-ar', '16000', '-ac', '1']}, # 16kHz mono audio required for NeMo tools
    'paths': {
        "home": "./_audio",
    },
    'outtmpl': '%(title)s.%(ext)s',
    'keepvideo': True,
    'audio-channels': 1,
    'downloader': 'ffmpeg',
    'hls-use-mpegts': True,
    'restrictfilenames': True,
}

def logInfo(URL):
    """
    Logs metadata about the video and saves it to a json file.

    Args:
        URL: The URL of the video to be logged.

    Returns:
        None
    """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(URL, download=False)

        filename = yt_dlp.utils.sanitize_filename(info['title'], restricted=True)

        # The Senate Video Player does not provide the title of the hearing, so we have to provide a unique identifier
        if 'm3u8' in info['original_url']:
            streamId = re.search("/([a-zA-Z]+[0-9]{6}\w*)/", info['original_url']).group(1)
            # print(streamId)
            if streamId:
                filename = info['title'] = streamId

        print(f"[LOG] getAudio - Saving info to info/{filename}.json")

        if not os.path.exists('./_info'):
            os.mkdir(path='./_info')

        with open(f'_info/{filename}.json', 'w') as f:
            f.write(json.dumps(ydl.sanitize_info(info)))

        print(f"[LOG] getAudio - Saved info to file")

        return f'_info/{filename}.json', filename

def downloadVideo(INFO_FILE):
    """
    Downloads a video using the provided info file, using the options configured above

    The video is downloaded in the './_audio' directory.

    Args:
        INFO_FILE: The path to the info file generated by logInfo.

    Returns:
        None
    """
    if not os.path.exists('./_audio'):
        os.mkdir(path='./_audio')

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download_with_info_file(INFO_FILE)

        if error_code:
            raise Exception(f"[ERROR] getAudio - Download failed with error code {error_code}")

def prepareAudio(URL):
    """
    Logs metadata about the video and downloads the audio to the '_audio' directory.
    
    Args:
        URL: The URL of the video to be downloaded.
    
    Returns:
        The title of the video, which is used as the filename of the downloaded audio.
    """
    info_file, contentId = logInfo(URL)
    # downloadVideo(info_file)
    return contentId

if __name__ == '__main__':
    prepareAudio(URL)