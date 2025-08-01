import yt_dlp
import json

ydl_opts = {
    'extract_flat': True,
    'dump_single_json': True,
    'quiet': True,
    'no_warnings': True,
}

# Use the channel's videos URL (either format works)
# channel_url = 'https://www.youtube.com/@GDiesen1/videos'
channel_url = "https://www.youtube.com/@WolfgangWeeUncut/videos"
# or: 'https://www.youtube.com/channel/CHANNEL_ID/videos'

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(channel_url, download=False)
    
    with open('channel_metadata_wee.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    
    print(f"Extracted {len(info['entries'])} videos from channel")