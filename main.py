from yt_dlp import YoutubeDL
import whisperx
import replicate
import os
import re
import time
from dotenv import load_dotenv
import requests
import math
import subprocess
import duckdb
import sqlite3
import json
import yaml
import kuzu
from pathlib import Path
from backbone import Backbone, Node, NodeMatch, Edge, EdgeMatch, Property, PropertyType, Config, Ontology # type: ignore
from pprint import pprint

load_dotenv()


class Podcast:
    def __init__(self, youtube_channel_url: str, backbone: Backbone = None, fetch_playlist: bool = False):
        self.backbone = backbone

        # set default values
        self.properties = {}
        self.properties["title"] = None
        self.properties["description"] = None
        self.properties["channel_meta_json"] = None
        self.properties["youtube_channel_url"] = youtube_channel_url

        self._meta = None
        self._loaded = False
        self._episodes = []

        self.read_backbone()

        if fetch_playlist and not self._loaded:
            self.fetch_playlist()

        self.load_episodes()

    def load_episodes(self):
        if not self._meta:
            return
        
        for entry in self._meta["entries"]:
            episode = PodcastEpisode(url=entry["url"], backbone=self.backbone)
            self._episodes.append(episode)

        print(f"Loaded {len(self._episodes)} episodes.")

    def get_episodes(self):
        return self._episodes

    def read_backbone(self):
        if self.backbone:
            podcasts = self.backbone.graph.get_node(
                NodeMatch(
                    label="Podcast",
                    where={"youtube_channel_url": self.properties.get("youtube_channel_url")},
                )
            )

            if podcasts and len(podcasts):
                self._load(podcasts[0])
            else:
                print("Podcast does not yet exist in Backbone")

       

    def _load(self, podcast: Node):
        print(f"Found existing podcast in Backbone: {podcast.properties.get("title").value}")

        self.properties["title"] = podcast.properties["title"].value
        self.properties["description"] = podcast.properties["description"].value
        self.properties["channel_meta_json"] = podcast.properties["channel_meta_json"].value

        self._meta = json.loads(podcast.properties["channel_meta_json"].value)
        
        self._loaded = True

    def fetch_playlist(self):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'dump_single_json': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            channel_meta = ydl.extract_info(self.properties.get("youtube_channel_url"), download=False)

            self.properties["title"] = channel_meta.get("title")
            self.properties["description"] = channel_meta.get("description")
            self.properties["channel_meta_json"] = json.dumps(channel_meta)

            self._meta = channel_meta # shorthand

            # Print some key metadata
            print(f"Channel title: {channel_meta.get('channel')}")
            print(f"Subscribers: {channel_meta.get('channel_follower_count')}")
            print(f"Description: {channel_meta.get('description')}")

        self._commit()

    def _commit(self):
        """Commit data to Backbone"""

        if not self.backbone:
            return
        
        podcast = Node(
            label="Podcast",
            # TODO: shouldn't be necessary to set key name, since it's redundant in Property itself. Could instead be a List[Property]
            properties={
                "youtube_channel_url": Property(
                    # TODO: name is misleading, should instead be KEY / VALUE
                    name="youtube_channel_url",
                    value=self.properties.get("youtube_channel_url"),
                    type=PropertyType.STRING
                ),
                "title": Property(
                    name="title",
                    value=self.properties.get("title"),
                    type=PropertyType.STRING
                ),
                "description": Property(
                    name="description",
                    value=self.properties.get("description"),
                    type=PropertyType.STRING
                ),
                "channel_meta_json": Property(
                    name="channel_meta_json",
                    value=self.properties.get("channel_meta_json"),
                    type=PropertyType.STRING
                ),
            }
        )
        saved = self.backbone.graph.add_node(podcast)
        print(f"Saved")
    
class PodcastEpisode:
    def __init__(self, url: str, backbone: Backbone = None):
        self._node = None
        self.url = url
        self.backbone = backbone

        self.read_backbone()

        if not self._node:
            # need to register a node
            node = Node(
                label="PodcastEpisode",
                properties={
                    "url": Property(
                        name="url",
                        value=self.url,
                        type=PropertyType.STRING
                    )
                }
            )
            self._node = self.backbone.graph.add_node(node)


    def read_backbone(self):
        if self.backbone:
            episodes = self.backbone.graph.get_node(
                NodeMatch(
                    label="PodcastEpisode",
                    where={"url": self.url},
                )
            )
            if episodes and len(episodes):
                node = episodes[0]
                self._node = node
    

    def post_hook(self, info):
        if info["status"] == "finished":

            self._filename = info["info_dict"]["_filename"]
            self._description = info["info_dict"]["description"]
            self._comment_count = info["info_dict"]["comment_count"]
            self._duration = info["info_dict"]["duration"]
            self._display_id = info["info_dict"]["display_id"]
            self._upload_date = info["info_dict"]["upload_date"]
            self._title = info["info_dict"]["title"]
            self._uploader = info["info_dict"]["uploader"]

            print(f"filename: {self._filename}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"description: {self._description}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"comment_count: {self._comment_count}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"duration: {self._duration}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"display_id: {self._display_id}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"upload_date: {self._upload_date}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"title: {self._title}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"
            print(f"uploader: {self._uploader}") # eg "downloads/Daniel Davis： Trump's Threats Against Russia Backfire [Fkqd1bJqaCU].m4a"

    def get_filename(self):
        """Extract filename from the full path stored in post_hook"""
        if hasattr(self, '_filename') and self._filename:
            f =  os.path.basename(self._filename)
            print(f"filename only: {f}")
            return f
        return None
    
    def set_transcript(self, transcript: dict):
        self._node.properties["transcript"].value = json.dumps(transcript)

        updated_node = self.backbone.graph.update_node_by_primary_key(
            label="PodcastEpisode",
            primary_key_name="url",
            primary_key_value=self.url,
            update_properties={
                "transcript": json.dumps(transcript)
            }
        )

        print(f"updated_node: {updated_node}")
        self._node = updated_node

    def get_transcript(self) -> dict:
        transcript_json = self._node.properties["transcript"].value
        print(type(transcript_json))
        if transcript_json:
            return json.loads(transcript_json)
        return None


class Pipeline:
    def __init__(self, backbone: Backbone, podcast: Podcast, max_episodes: int = 3):
        self.backbone = backbone
        self.podcast = podcast
        self.max_episodes = max_episodes
 
    def run(self):

        # get list of episodes from podcast (already stored in metadata)
        # create PodcastEpisode class
        # create Person's (host, guests)
        # get full metadata of episode
        # download audio of episode as wav (to tmp file)
        # transcribe using Replicate API (needs file hosted with ngrok)
        # clean up and process transcription & diarization using LLM
        # create Paragraph's
        # store Nodes
        # store Edges

        episodes = self.podcast.get_episodes()
        print(f"Pipeline episodes: {len(episodes)}")

        for episode in self.podcast.get_episodes()[:1]:
            self.run_episode_tasks(episode)

    def run_episode_tasks(self, episode: PodcastEpisode):
            
            # download
            self.download_episode(episode) # post-hook stores meta

            # transcribe
            self.transcribe_episode(episode)

            # clean up transcription / diarization
            self.cleanup_transcription(episode)

    def cleanup_transcription(self, episode: PodcastEpisode):
        turns = self._combine_turns(episode)

        print(len(turns))

    def _combine_turns(self, episode: PodcastEpisode) -> list[dict]:
        transcript = episode.get_transcript()
        turns = transcript["segments"]

        combined_turns = []
        prev_turn = None
        current_texts = []
        for turn in turns:
            print(f"{turn["speaker"]} start:{turn["start"]} end:{turn["end"]}")
            
            # first round
            if not prev_turn:
                prev_turn = turn
                start_of_combined_turn = turn["start"]
                current_texts.append(turn["text"])
                continue

            # same speaker, continue
            if prev_turn["speaker"] == turn["speaker"]:
                prev_turn = turn
                current_texts.append(turn["text"])
                continue

            # new speaker, record combined turn
            else:

                # store combined turn
                combined_turns.append({
                    "end": prev_turn["end"],
                    "speaker": prev_turn["speaker"],
                    "start": start_of_combined_turn,
                    "text" : " ".join(current_texts)
                })

                # mark new start
                start_of_combined_turn = turn["start"]
                current_texts = []
                prev_turn = turn

        # last turn
        combined_turns.append({
            "end": prev_turn["end"],
            "speaker": prev_turn["speaker"],
            "start": start_of_combined_turn,
            "text" : " ".join(current_texts)
        })

        for combined_turn in combined_turns:
            print(f"\n---\n{combined_turn["speaker"]} start:{combined_turn["start"]} end:{combined_turn["end"]}")
            print(combined_turn["text"])

        return combined_turns

    def transcribe_episode(self, episode: PodcastEpisode):
        """
        Transcribe using Replicate interference.
        
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for progress
        """

        if episode.get_transcript():
            print("Transcript already exists. Skipping!")
            return

        output = replicate.run(
            "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
            input={
                "debug": False,
                "vad_onset": 0.5,
                "audio_file": f"{os.getenv("NGROK_URL")}/{episode.get_filename()}",
                "batch_size": 64,
                "vad_offset": 0.363,
                "diarization": True,
                "temperature": 0,
                "align_output": True,
                "max_speakers": 3, # TODO: set dynamically
                "min_speakers": 1,
                "huggingface_access_token": os.getenv("HUGGINGFACE_TOKEN"),
                "language_detection_min_prob": 0,
                "language_detection_max_tries": 5
            }
        )

        episode.set_transcript(output)


   
    def download_episode(self, episode):
        ydl_opts = {
            "paths": {"home": "downloads"},
            "format": "m4a/bestaudio/best",
            "cookiesfrombrowser": ("brave", "default", "BASICTEXT"),
            # "cookiefile": "youtube.cookie"
            "postprocessor_hooks": [episode.post_hook]
        }

        print(f"download ydl_opts {ydl_opts}")

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([episode.url])
            print(f"error_code: {error_code}")
        


class Planning:
    """

    INPUT is a YT channel url
    
    (For each step, store data directly in Kuzu)

    - Create PODCAST item, which contains YT url, title, author, list of episodes
        - If already exists, continue
    - CREATE EPISODE item, from each episode in PODCAST episode list
        - If already exists, skip? Or check if all content is present?
    - @task Download episode from YT as audio file m4a
        - INPUT is direct youtube url to episode
    - @task Get meta of episode (title, people, date, length, description, etc.)
    - @task Convert m4a to wav
         - Unless we can download directly as wav
    - @task Transcribe & diarize audio using Replicate
    - @task Process transcription, clean up using LLM

    - Store all of above to Kuzu:
        - Podcast
            |name
            |host
            |etc
        
        - PodcastEpisode
            |title
            |num
            |guest(s)
            |date
            |description
            |length
            |url
            |_t11n_raw (raw json from replicate)
            |_t11n_clean (cleaned up transcription, into paragraphs)
            |
            |>> PodcastEpisode-CONTENT->Paragraph(first)
            |>> Person-IS_GUEST_ON->PodcastEpisode
            |>> Person-IS_HOST_OF->PodcastEpisode
        
        - Paragraph / Utterance / Speech
            |text
            |embedding
            |timestamp (start/end)
            |
            |>> Paragraph-NEXT->Paragraph rel.
            |>> Paragraph-SPOKEN_BY->Person

        - Person
            |name
            |dbpedia_uri
            |profession
            |
            |>> Paragraph-SPOKEN_BY->Person
            
    We now have a LEXICAL graph.

    - Then, start to find entities, relationships:
        - Person
        - Place
        - People (eg. amerikanerene, russerne, ukrainere)
        - Date (Year, specific date)
        - Organisation
        - Country
        - Concept (våpenhvile, energi)
        
        DBPedia Ontology (add dbpedia_uri (eg. http://dbpedia.org/resource/Odessa) to everything)
        + Country
        + City
        + Place
        + Person
        + Organisation
        + NaturalResource
        + MilitaryConflict
        + Treaty
        + Event
        + Speech / Utterance
        + Podcast
        + Agreement
        + Newspaper
        + Politician

        - isPartOf
        - isClaimedBy
        - hasMilitaryPresence
        - isInvolvedInConflict
        - hasEconomicInterest
        - proposedAgreement
        - participatedIn
        - criticised
        - supported
        - occurredIn
        - origin
        - transportedThrough
        - includes
        - hasPosition
        - generatedStatement
        - hasSource
        - spokenBy
        - hasTimestamp
        - hasConfidence
    """


def main():

    with open("podcast-ontology.yaml", "r") as yaml_file:
        ontology_dict = yaml.safe_load(yaml_file)   

    config = Config(kuzu_data_path=str(Path(__file__).parent / "data/podcast.kuzu"))
    ontology = Ontology(ontology_dict)
    backbone = Backbone(config)
    # backbone.clear_graph()
    backbone.set_ontology(ontology)
    print(backbone.get_ontology().as_yaml())

    podcast = Podcast(
        youtube_channel_url="https://www.youtube.com/@GDiesen1/videos", 
        backbone=backbone
    )
    # podcast.fetch_playlist()

    pipeline = Pipeline(backbone, podcast)

    pipeline.run()



if __name__ == "__main__":
    main()
