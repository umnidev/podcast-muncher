import yt_dlp
import json
from pprint import pprint

# Configure yt-dlp options
ydl_opts = {
    'extract_flat': False,  # Get full metadata
    'quiet': True,
    'no_warnings': True,
    "cookiesfrombrowser": ("brave", "default", "BASICTEXT"),
    # "cookiefile": "youtube.cookie"
}

# Use any of these URL formats:
url = 'https://www.youtube.com/@GDiesen1'  # Handle format
# or: 'https://www.youtube.com/channel/UCXXXXXXX'  # Channel ID format
# or: 'https://www.youtube.com/user/Username'  # Legacy username format

url = "https://www.youtube.com/watch?v=5ySKZhY3KmI"

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    pprint(info)

    
    # Save to file
    with open('channel_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    
    # Print some key metadata
    print(f"Channel Name: {info.get('channel')}")
    print(f"Subscribers: {info.get('channel_follower_count')}")
    print(f"Description: {info.get('description')[:100]}...")