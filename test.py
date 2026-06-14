from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse

def extract_transcript(yt_url):

    parsed = urlparse(yt_url)

    if "youtu.be" in parsed.netloc:
        video_id = parsed.path[1:]
    else:
        video_id = yt_url.split("v=")[1].split("&")[0]

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    text = " ".join([item.text for item in transcript])

    return text


text = extract_transcript(
    "https://youtu.be/HFfXvfFe9F8?si=8xAVNPlo2G0WNtGs"
)

print(text)
print(len(text))