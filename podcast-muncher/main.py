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
from neo4j import GraphDatabase
from llm import SmoothTranscription
 
# memgraph running on host
memgraph = GraphDatabase.driver(uri="bolt://localhost:7687", auth=("", ""))
memgraph.verify_connectivity()

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
    def __init__(self, url: str):
        self.url = url
        self._properties = {}
        self.PodcastEpisodes = []

    def fetch_meta(self):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'dump_single_json': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Fetching meta for {self.url}...")

            meta = ydl.extract_info(self.url, download=False)
            if not meta:
                logging.error(f"Failed to fetch meta for {self.url}")
                return
            
            self._properties["meta"] = meta
            self._properties["title"] = meta.get("title")
            self._properties["description"] = meta.get("description")

    def load_episodes(self) -> list:
        self.PodcastEpisodes = []

        if not self._properties.get("meta"):
            logging.warning("No meta, can't load episodes.")
            return

        entries = self._properties["meta"].get("entries", [])
        logging.info(f"Loading {len(entries)} episodes from meta.")

        for entry in entries:
            episode = PodcastEpisode(url=entry.get("url"), podcast=self, properties={
                "title": entry.get("title"),
                "description": entry.get("description")
            })
            self.PodcastEpisodes.append(episode)

        return self.PodcastEpisodes
     
    def save(self):
        records, summary, keys = memgraph.execute_query(
            """
            MERGE (p:Podcast {url: $url}) 
            SET p.title = $title, 
                p.description = $description, 
                p.meta = $meta 
            RETURN p
            """,
            url=self.url,
            **self._properties
        )
        logging.info(f"Merged Podcast node.")

    def load(self):
        logging.info(f"Loading Podcast node for URL: {self.url}")
        records, summary, keys = memgraph.execute_query(
            """
            MATCH (p:Podcast) 
            WHERE p.url = $url 
            RETURN p 
            LIMIT 1
            """,
            url=self.url
        )
        if records and len(records):
            self._properties = records[0]["p"]._properties


class PodcastEpisode:
    def __init__(self, url: str, podcast: Podcast, properties: dict = {}):
        self.url = url
        self.podcast = podcast
        self._properties = properties

    def save(self):
        if not self._properties:
            logging.warning("No properties to save.")
            return
        
        # Build SET clause dynamically - use 'pe' to match the MERGE variable
        set_clauses = [f"pe.{key} = ${key}" for key in self._properties.keys()]
        set_clause = ", ".join(set_clauses)
    
        query = f"""
        MERGE (pe:PodcastEpisode {{url: $url}}) 
        SET {set_clause}
        WITH pe
        MATCH (p:Podcast {{url: $podcast_url}})
        MERGE (p)-[:HAS_EPISODE]->(pe)
        RETURN pe
        """
        
        records, summary, keys = memgraph.execute_query(
            query,
            url=self.url,
            podcast_url=self.podcast.url,
            **self._properties
        )

        logging.info(f"Merged PodcastEpisode node.")

    def download(self):
        """Download episode using yt-dlp. Save to file."""

        ydl_opts = {
            "paths": {"home": "downloads"},
            "format": "m4a/bestaudio/best",
            "cookiesfrombrowser": ("brave", "default", "BASICTEXT"),
            # "cookiefile": "youtube.cookie"
            "postprocessor_hooks": [self.post_hook]
        }

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([self.url])

    def transcribe(self):
        """
        Transcribe using Replicate interference.
        
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for progress
        """

        if self._properties.get("transcript"):
            logging.info("Transcript already exists. Skipping!")
            return

        logging.info(f"Transcribing {self._properties.get('title')}...")

        transcript = replicate.run(
            "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
            input={
                "debug": False,
                "vad_onset": 0.5,
                "audio_file": f"{os.getenv("NGROK_URL")}/{self.get_filename()}",
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

        if not transcript:
            raise Exception("Transcription failed.")

        self._properties["transcript"] = transcript

    def post_process_transcription(self):
        """
        Post-process transcription to combine speaker turns into paragraphs,
        and clean up the text using a language model.
        This will also add summaries to each paragraph.
        """
        if not self._properties.get("transcript"):
            raise Exception("No transcript to post-process.")
        
        smooth_operator = dspy.ChainOfThought(SmoothTranscription)

        paragraphs = []
        for turn in self._combine_turns():
            pred = smooth_operator(
                speech=turn["speech"], 
                context=f"Episode description: {self._properties.get('description', '')}",
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

        self._properties["paragraphs"] = paragraphs
    

    def post_hook(self, info):
        """Post-hook for yt-dlp to store metadata after download."""
        if info.get("status") == "finished":
            self._properties.update({
                "filename": info["info_dict"]["_filename"],
                "duration": info["info_dict"]["duration"],
                "date": datetime.strptime(info["info_dict"]["upload_date"], "%Y%m%d").date(),
                "description": info["info_dict"]["description"],
                "display_id": info["info_dict"]["display_id"]
            })
            logging.info(f"Downloaded successfully. {self._properties}")

    def get_filename(self):
        """Extract filename from the full path stored in post_hook"""
        filename = self._properties.get("filename")
        return os.path.basename(filename)

    def _combine_turns(self) -> list[dict]:
        """Combine speaker turns into one full turn."""

        logging.info("Combining speaker turns into paragraphs...")

        transcript = self._properties.get("transcript")
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

    
    # def _sync(self):
    #     """
    #     A method to sync the data in the class 
    #     with that in the Backbone graph.

    #     Should be idempotent.

    #     Should check if node exists in Backbone.
    #         If exists, should update the class.
    #         If not exists, should create immediately.
    #     If already inited, ie. self._node is not None,
    #         then any unsynced parameters between class
    #         backbone should be synced, class overwriting 
    #         backbone. 
    #     Should also handle edges. 

    #     ###

    #     - it's tricky to mirror three things: real-world text, classes, and graph nodes/edges
    #     - perhaps this is not the best approach
    #     - the main problem arises from needing to be idempotent; both creating and handling
    #       existing, and syncing both ways. 
    #     - would be a lot easier if we did one-way street: churning through data and writing
    #       into backbone, and not dealing with classes created from existing nodes/edges in bb.
    #     - Podcast class is the only one that updates in the real world: there are more episodes
    #       added to it. Episodes, once created, do not change, nor does its content.
    #     - In other words, Podcast class could be read from backbone, and its meta re-fetched,
    #       in order to discover new episodes. Everything else, ie. PodcastEpisode classes, 
    #       Paragraph classes, etc. are created just in order to process and store data, once.
    #     - Thus, we do not need to worry about retriving existing Paragraphs, for example, but
    #       instead we create them, always create.
    #     - This greatly simplifies our handling.  
    #     """
    #     pass

        


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


# class SmoothTranscription(dspy.Signature):
#     """
#     Minimally clean transcription. Make it more legible, but keep it as close to original as possible.
#     Fix grammatical errors and transcription errors. 
#     Make sure NOT to leave out ANY significant detail from the transcript, ie. names, meanings.

#     Also add short- and long summaries, in the voice and perspective of the SPEAKER.
#     """

#     speech: str = dspy.InputField()
#     context: str = dspy.InputField()
#     text: str = dspy.OutputField()
#     summary: str = dspy.OutputField(desc="One sentence summary of text (as if spoken by the SPEAKER).")
#     long_summary: str = dspy.OutputField(desc="One paragraph summary of text (as if spoken by the SPEAKER).")

class Pipeline:
    def __init__(self, podcast: Podcast, max_episodes: int = 3):
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

        episodes = self.podcast.PodcastEpisodes
        if not episodes:
            episodes = self.podcast.load_episodes()
            
        logging.info(f"Pipeline episodes: {len(episodes)}. Handling {self.max_episodes} episodes.")

        for episode in episodes[:self.max_episodes]:
            self.run_episode_tasks(episode)

    def run_episode_tasks(self, episode: PodcastEpisode):

            try:
                # download
                episode.download()

                # transcribe
                episode.transcribe()

                # clean up transcription / diarization
                episode.post_process_transcription()

                # save episode to graph
                episode.save()
            except Exception as e:
                logging.error(f"Error processing episode {episode.url}: {e}")
                logging.warning("Episode was not saved to graph.")


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


def init_memgraph():
    """
    Initialize Memgraph database with constraints and indexes.
    """
    with memgraph.session() as session:
        # Create constraint for Podcast
        session.run("CREATE CONSTRAINT ON (p:Podcast) ASSERT p.url IS UNIQUE")
        logging.info("Created constraint for Podcast.")

        # Create constraint for PodcastEpisode
        session.run("CREATE CONSTRAINT ON (pe:PodcastEpisode) ASSERT pe.url IS UNIQUE")
        logging.info("Created constraint for PodcastEpisode.")

        # Create index for Person
        session.run("CREATE INDEX ON :Person(name)")
        logging.info("Created index for Person.")

def clear_memgraph():
    """
    Clear all nodes and edges in Memgraph database.
    """
    with memgraph.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        logging.info("Cleared all nodes and edges in Memgraph.")

# def main():

#     with open("podcast-ontology.yaml", "r") as yaml_file:
#         ontology_dict = yaml.safe_load(yaml_file)   

#     config = Config(kuzu_data_path=str(Path(__file__).parent / "data/podcast.kuzu"))
#     ontology = Ontology(ontology_dict)
#     backbone = Backbone(config)
#     # backbone.clear_graph()
#     backbone.set_ontology(ontology)
#     logging.debug(backbone.get_ontology().as_yaml())

#     # quit()
#     # episode = PodcastEpisode(url="https://www.youtube.com/watch?v=Fkqd1bJqaCU", backbone=backbone)
#     # for segment in episode.get_paragraphs():
#     #     print(f"{segment["speaker"]}: {segment["summary"]}")
#     # quit()

#     podcast = Podcast(
#         url="https://www.youtube.com/@GDiesen1/videos", 
#         backbone=backbone
#     )
#     logging.info(f"Podcast initiated: {podcast} {podcast.url}")

#     quit()

#     pipeline = Pipeline(backbone, podcast)
#     pipeline.run_podcast()

    
#     # pipeline.run_episode(url="")


def main():
    # init_memgraph()
    # clear_memgraph()

    podcast = Podcast(url="https://www.youtube.com/@GDiesen1/videos")
    podcast.fetch_meta()
    podcast.save()
    # podcast.load()
    episodes = podcast.load_episodes()
    logging.info(f"Podcast episodes: {len(episodes)}")

    pipeline = Pipeline(podcast, max_episodes=4)
    pipeline.run_podcast()



if __name__ == "__main__":
    main()
