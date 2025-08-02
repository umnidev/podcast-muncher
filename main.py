from yt_dlp import YoutubeDL
import replicate
import os
import re
import time
from dotenv import load_dotenv
import requests
import math
import subprocess
import json
import yaml
from pathlib import Path
from backbone import Backbone, Node, NodeMatch, Edge, EdgeMatch, Property, PropertyType, Config, Ontology # type: ignore
from pprint import pprint
import dspy
from datetime import datetime, date
import logging

logging.basicConfig(level=logging.INFO, format="{levelname} - {message}", style="{")

load_dotenv()

lm = dspy.LM("openai/gpt-4.1-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)


# TODO:
# - Connect Podcast-HAS_EPISODE->PodcastEpisode
# - Connect PodcastEpisode - FIRST_PARAGRAPH -> Paragraph
# - Connect Paragraph -> NEXT_PARAGRAPH -> Paragraph
# - Connect Paragraph -> EXPRESSED_BY -> Person

class Podcast:
    def __init__(self, url: str, backbone: Backbone):
        self.url = url
        self._backbone = backbone
        
        self._node = None
        self._meta = None
        self.PodcastEpisodes = []

        self._load_node()

        if not self._node:
            self._create_node()

        # always fetch latest
        self.fetch_meta()

        # self.load_episodes()
        self.thin_load_episodes()

    def _create_node(self):
        logging.info("Creating fresh node")
        node = Node(
            label="Podcast",
            properties={
                "url": Property(
                    name="url",
                    value=self.url,
                    type=PropertyType.STRING
                )
            }
        )
        self._node = self._backbone.graph.add_node(node)
        logging.info("Created Podcast node.")

    def _load_node(self):
        res = self._backbone.graph.get_node(
            NodeMatch(
                label="Podcast",
                where={"url": self.url},
            )
        )

        if res and len(res):
            self._node = res[0]
            self._meta = json.loads(self._node.properties["meta_json"].value)
       
    def fetch_meta(self):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'dump_single_json': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            meta = ydl.extract_info(self.url, download=False)

            self._meta = meta

            logging.info(f"Fetched meta")

        self._update_backbone({
            "title": meta.get("title"),
            "description": meta.get("description"),
            "meta_json": json.dumps(meta),
        })

    # def thin_load_episodes(self):
    #     if not self._meta:
    #         logging.warning("No meta, can't load episode.")
    #         return
        
    #     entries = self._meta.get("entries")
    #     logging.info(f"Entries: {len(entries)}")

    #     # so, simplest way:
    #     # - just load episodes locally, ie. no PodcastEpisode class
    #     # - THIS SUCKS!!


    def load_episodes(self):
        if not self._meta:
            logging.warning("No meta, can't load episode.")
            return
        
        entries = self._meta.get("entries")
        logging.info(f"Entries: {len(entries)}")
  
        for entry in self._meta.get("entries"):
            episode = PodcastEpisode(url=entry.get("url"), backbone=self._backbone)
            self.PodcastEpisodes.append(episode)

            node_results = self._backbone.graph.get_node(
                NodeMatch(
                    label="PodcastEpisode",
                    where={"url": episode.url}
                )
            )
            ep_node = node_results[0]

            # ensure edge between nodes
            self._backbone.graph.add_edge(Edge(
                label="HAS_EPISODE",
                from_node=self._node,
                to_node=ep_node
            ))

        logging.info(f"Loaded {len(self.PodcastEpisodes)} episodes.")

    def get_episodes(self):
        return self.PodcastEpisodes

    def _update_backbone(self, update_properties: dict):
        self._node = self._backbone.graph.update_node_by_primary_key(
            label="Podcast",
            primary_key_name="url",
            primary_key_value=self.url,
            update_properties=update_properties
        )
        logging.debug(f"Podcast: updated_node: {self._node}")

class PodcastEpisode:
    def __init__(self, url: str, backbone: Backbone):
        self.url = url
        self._backbone = backbone

        self._node = None
        self._properties = {}
        self.Paragraphs = []

    def save(self):
        if not self._node:
            self._load_node()
    
        if not self._node:
            self._create_node()
        
        self._node = self._backbone.graph.update_node_by_primary_key(
            label="PodcastEpisode",
            primary_key_name="url",
            primary_key_value=self.url,
            update_properties=self._properties
        )

    def _create_node(self):
        self._node = self._backbone.graph.add_node(Node(
            label="PodcastEpisode",
            properties={
                "url": Property(
                    name="url",
                    value=self.url,
                    type=PropertyType.STRING
                )
            }
        ))

    def _load_node(self):
        res = self._backbone.graph.get_node(
            NodeMatch(
                label="PodcastEpisode",
                where={"url": self.url},
            )
        )
        if res and len(res):
            self._node = res[0]

            # set properties from node, but don't overwrite existing on class
            existing_properties = self._properties
            node_properties = {key: prop.value for key, prop in self._node.properties.items()}
            self._properties = node_properties | existing_properties

    def post_hook(self, info):
        if info.get("status") == "finished":
            self._properties.update({
                "title": info["info_dict"]["_filename"],
                "duration": info["info_dict"]["duration"],
                "date": datetime.strptime(info["info_dict"]["upload_date"], "%Y%m%d").date(),
                "description": info["info_dict"]["description"],
                "display_id": info["info_dict"]["display_id"]
            })

    def get_filename(self):
        """Extract filename from the full path stored in post_hook"""
        if hasattr(self, '_filename') and self._filename:
            f =  os.path.basename(self._filename)
            return f
    
    def set_transcript(self, transcript: dict):
        self._properties.update({
            "transcript": json.dumps(transcript)
        })

    def get_transcript(self) -> dict:
        t_json = self.properties.get("transcript")
        return json.loads(t_json) if t_json else None
    
    def get_description(self) -> str:
        return self._properties.get("descriptioon")
    
    def set_paragraphs(self, paragraphs: dict):
        """Called from Pipeline, ie. always with fresh data."""

        self._properties.update({
            "paragraphs_json": json.dumps(paragraphs)
        })

    def get_paragraphs(self):
        return self.Paragraphs

    def _sync(self):
        """
        A method to sync the data in the class 
        with that in the Backbone graph.

        Should be idempotent.

        Should check if node exists in Backbone.
            If exists, should update the class.
            If not exists, should create immediately.
        If already inited, ie. self._node is not None,
            then any unsynced parameters between class
            backbone should be synced, class overwriting 
            backbone. 
        Should also handle edges. 

        ###

        - it's tricky to mirror three things: real-world text, classes, and graph nodes/edges
        - perhaps this is not the best approach
        - the main problem arises from needing to be idempotent; both creating and handling
          existing, and syncing both ways. 
        - would be a lot easier if we did one-way street: churning through data and writing
          into backbone, and not dealing with classes created from existing nodes/edges in bb.
        - Podcast class is the only one that updates in the real world: there are more episodes
          added to it. Episodes, once created, do not change, nor does its content.
        - In other words, Podcast class could be read from backbone, and its meta re-fetched,
          in order to discover new episodes. Everything else, ie. PodcastEpisode classes, 
          Paragraph classes, etc. are created just in order to process and store data, once.
        - Thus, we do not need to worry about retriving existing Paragraphs, for example, but
          instead we create them, always create.
        - This greatly simplifies our handling.  
        """

        


class Paragraph:
    """Each paragraph of text, from a single speaker."""
    def __init__(self, meta: dict, backbone: Backbone, podcast_episode: PodcastEpisode):
        self._meta = meta
        self._backbone = backbone
        self.PodcastEpisode = podcast_episode

        self._node = None


class Sentence:
    """Could be cool, but too complicated for now."""
    def __init__(self, text: str):
        pass


class SmoothTranscription(dspy.Signature):
    """
    Minimally clean transcription. Make it more legible, but keep it as close to original as possible.
    Fix grammatical errors and transcription errors. 
    Make sure NOT to leave out ANY significant detail from the transcript, ie. names, meanings.

    Also add short- and long summaries, in the voice and perspective of the SPEAKER.
    """

    speech: str = dspy.InputField()
    context: str = dspy.InputField()
    text: str = dspy.OutputField()
    summary: str = dspy.OutputField(desc="One sentence summary of text (as if spoken by the SPEAKER).")
    long_summary: str = dspy.OutputField(desc="One paragraph summary of text (as if spoken by the SPEAKER).")

class Pipeline:
    def __init__(self, backbone: Backbone, podcast: Podcast, max_episodes: int = 3):
        self._backbone = backbone
        self.podcast = podcast
        self.max_episodes = max_episodes
 
    def run_podcast(self):

        # get list of episodes from podcast (already stored in metadata)
        # create PodcastEpisode class
        # create Person's (host, guests)
        # get full metadata of episode
        # download audio of episode as wav (to tmp file)
        # transcribe using Replicate API (needs file hosted with ngrok)
        # clean up and process transcription & diarization using LLM
        # identify SPEAKER -> Person
        # create Paragraph's
        # store Nodes
        # store Edges

        episodes = self.podcast.get_episodes()
        logging.info(f"Pipeline episodes: {len(episodes)}")

        for episode in self.podcast.get_episodes()[:2]:
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

        smooth_operator = dspy.ChainOfThought(SmoothTranscription)

        paragraphs = []
        for turn in turns:
            pred = smooth_operator(
                speech=turn["speech"], 
                context=f"Episode description: {episode.get_description()}"
            )
            
            turn["text"] = pred.text
            turn["summary"] = pred.summary
            turn["long_summary"] = pred.long_summary

            print("----start-----")
            print(f"\nspeech: {turn["speech"]}")
            print(f"\ntext: {pred.text}")
            print(f"\nsummary: {pred.summary}")
            print(f"\nlong_summary: {pred.long_summary}")
            print("----end-----")

            paragraphs.append(turn)

        episode.set_paragraphs(paragraphs)


    def _combine_turns(self, episode: PodcastEpisode) -> list[dict]:
        """Combine speaker turns into one full turn."""

        transcript = episode.get_transcript()

        pprint(f"Transcript {transcript}")

        turns = transcript["segments"]

        combined_turns = []
        current_texts = []
        prev_turn = None
        for turn in turns:
            
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
                    "speech" : " ".join(current_texts)
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
            "speech" : " ".join(current_texts)
        })

        return combined_turns

    def transcribe_episode(self, episode: PodcastEpisode):
        """
        Transcribe using Replicate interference.
        
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for progress
        """

        if episode.get_transcript():
            logging.info("Transcript already exists. Skipping!")
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

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([episode.url])


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
    logging.debug(backbone.get_ontology().as_yaml())

    # quit()
    # episode = PodcastEpisode(url="https://www.youtube.com/watch?v=Fkqd1bJqaCU", backbone=backbone)
    # for segment in episode.get_paragraphs():
    #     print(f"{segment["speaker"]}: {segment["summary"]}")
    # quit()

    podcast = Podcast(
        url="https://www.youtube.com/@GDiesen1/videos", 
        backbone=backbone
    )
    logging.info(f"Podcast initiated: {podcast} {podcast.url}")

    quit()

    pipeline = Pipeline(backbone, podcast)
    pipeline.run_podcast()

    
    # pipeline.run_episode(url="")



if __name__ == "__main__":
    main()
