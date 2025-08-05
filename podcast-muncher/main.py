from functools import wraps
from yt_dlp import YoutubeDL
import replicate
from replicate.exceptions import ModelError
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
from pprint import pprint
import dspy
from datetime import datetime, date, timezone
import logging
from neo4j import GraphDatabase
from neo4j.time import DateTime
from llm import CleanTranscript, DefineSpeakers
import concurrent.futures

# memgraph running on host
memgraph = GraphDatabase.driver(uri="bolt://localhost:7687", auth=("", ""))
memgraph.verify_connectivity()

logging.basicConfig(
    level=logging.INFO, format="{levelname} - {message}", style="{"
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # If using httpx
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

load_dotenv()

lm = dspy.LM("openai/gpt-4.1-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)


def retry_on_failure(max_retries=3, delay=1.0, exponential_backoff=True):
    """Decorator to retry functions on failure."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        logging.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logging.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
            return None

        return wrapper

    return decorator


@retry_on_failure(max_retries=3, delay=1.0)
def memgraph_query(query: str, **params):
    """
    Execute a query against the Memgraph database.
    """
    records, summary, keys = memgraph.execute_query(query, **params)
    return records, summary, keys


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


    GRAPHICAL USER INTERFACE
    - showing podcast episodes
        - easy reading of paragraphs (and easy choice of one sentence, few sentences, full text)
    - search for topics, entities, where they are mentioned. return text snippets, with links
      to full text, or directly to podcast episode at timestamp.
    - semantic search, returning paragraphs that discuss search query

    Return objects:
    - Paragraph
        - text
        - one_sentence
        - few_sentences
        - timestamp (start, end)
        - word by word text (with timestamps)
        - reference to PodcastEpisode
        - reference to Person(s)
        - entities (list of entities mentioned in the paragraph)
        - triples (list of triples extracted from the paragraph, ie. edges)



    """


class Podcast:
    def __init__(self, url: str):
        self.url = url
        self._properties = {}
        self.PodcastEpisodes = []

    def fetch_meta(self):
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "dump_single_json": True,
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
            episode = PodcastEpisode(
                url=entry.get("url"),
                podcast=self,
                properties={
                    "title": entry.get("title"),
                    "description": entry.get("description"),
                },
            )
            episode.load()
            self.PodcastEpisodes.append(episode)

        return self.PodcastEpisodes

    def save(self):
        records, summary, keys = memgraph_query(
            """
            MERGE (p:Podcast {url: $url}) 
            SET p.title = $title, 
                p.description = $description, 
                p.meta = $meta,
                p.updated_at = datetime()
            RETURN p
            """,
            url=self.url,
            **{k: v for k, v in self._properties.items() if k != "url"},
        )
        logging.info(f"Merged Podcast node.")

    def load(self):
        logging.info(f"Loading Podcast node for URL: {self.url}")
        records, summary, keys = memgraph_query(
            """
            MATCH (p:Podcast) 
            WHERE p.url = $url 
            RETURN p 
            LIMIT 1
            """,
            url=self.url,
        )
        if records and len(records):
            self._properties = records[0]["p"]._properties
            return self


class Transcript:
    """Transcript node in the graph, representing the full transcript of a podcast episode."""

    def __init__(self, podcast_episode_url: str, properties: dict = {}):
        self._properties = properties
        self.podcast_episode_url = podcast_episode_url

    def save(self):
        """Save Transcript node to Memgraph and create relationship to PodcastEpisode."""

        # Build SET clause dynamically
        set_clause = ", ".join(
            [f"t.{key} = ${key}" for key in self._properties.keys()]
            + ["t.updated_at = datetime()"]
        )

        logging.info(f"Saving Transcript node for {self.podcast_episode_url}.")

        query = f"""
        MERGE (t:Transcript {{podcast_episode_url: $podcast_episode_url}})
        SET {set_clause}
        WITH t

        MATCH (pe:PodcastEpisode {{url: $podcast_episode_url}})
        MERGE (pe)-[:HAS_TRANSCRIPT {{updated_at: datetime()}}]->(t)
        RETURN t
        """
        records, summary, keys = memgraph_query(
            query,
            **self._properties,
            podcast_episode_url=self.podcast_episode_url,
        )

        logging.info(f"Saved Transcript node for {self.podcast_episode_url}")

    def load(self) -> "Transcript":
        records, summary, keys = memgraph_query(
            """
            MATCH (t:Transcript {podcast_episode_url: $podcast_episode_url}) 
            RETURN t 
            LIMIT 1
            """,
            podcast_episode_url=self.podcast_episode_url,
        )
        if records and len(records):
            self._properties = records[0]["t"]._properties
            return self


class PodcastEpisode:
    def __init__(self, url: str, podcast: Podcast, properties: dict = {}):
        self.url = url
        self.podcast = podcast
        self._properties = properties
        self._save_queue = []
        self.Paragraphs = []

    def save(self):
        if not self._properties:
            logging.warning("No properties to save.")
            return

        # Build SET clause dynamically - use 'pe' to match the MERGE variable
        set_clause = ", ".join(
            [f"pe.{key} = ${key}" for key in self._properties.keys()]
            + ["pe.updated_at = datetime()"]
        )

        query = f"""
        MERGE (pe:PodcastEpisode {{url: $url}}) 
        SET {set_clause}
        WITH pe

        MATCH (p:Podcast {{url: $podcast_url}})
        MERGE (p)-[:HAS_EPISODE {{updated_at: datetime()}}]->(pe)
        RETURN pe
        """
        logging.info(f"Saving PodcastEpisode node for URL: {self.url}")

        records, summary, keys = memgraph_query(
            query,
            url=self.url,
            podcast_url=self.podcast.url,
            **{k: v for k, v in self._properties.items() if k != "url"},
        )

        # process save queue (items which have this PodcastEpisode as dependency)
        for item in self._save_queue:
            item.save()
        self._save_queue = []  # Clear the queue after saving

    def load(self):
        logging.info(f"Loading PodcastEpisode node for URL: {self.url}")
        records, summary, keys = memgraph_query(
            """
            MATCH (pe:PodcastEpisode) 
            WHERE pe.url = $url 
            RETURN pe 
            LIMIT 1
            """,
            url=self.url,
        )
        if records and len(records):
            self._properties = records[0]["pe"]._properties
            return self

    def download(self, overwrite: bool = False):
        """Download episode using yt-dlp. Save to file."""

        transcript = Transcript(podcast_episode_url=self.url).load()
        if transcript and not overwrite:
            logging.info("Already downloaded. Skipping!")
            return

        ydl_opts = {
            "paths": {"home": "downloads"},
            "format": "m4a/bestaudio/best",
            "cookiesfrombrowser": ("brave", "default", "BASICTEXT"),
            # "cookiefile": "youtube.cookie"
            "postprocessor_hooks": [self.post_download_hook],
            "quiet": True,
            "noprogress": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([self.url])
            logging.info(
                f"Downloaded {self._properties.get('title')} with error code: {error_code}"
            )

    def transcribe(self, overwrite: bool = False):
        """
        Transcribe using Replicate interference.
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for model details
        """

        transcript = Transcript(podcast_episode_url=self.url).load()
        if transcript and not overwrite:
            logging.info("Transcript already exists. Skipping!")
            return

        logging.info(f"Transcribing {self._properties.get('title')}...")

        model = replicate.models.get(os.getenv("TRANSCRIPTION_MODEL"))
        version = model.versions.get(os.getenv("TRANSCRIPTION_MODEL_VERSION"))
        prediction = replicate.predictions.create(
            version=version,
            input={
                "debug": False,
                "vad_onset": 0.5,
                "audio_file": f"{os.getenv("NGROK_URL")}/{self._get_filename()}",
                "batch_size": 64,
                "vad_offset": 0.363,
                "diarization": True,
                "temperature": 0,
                "align_output": True,
                "max_speakers": 3,  # TODO: set dynamically
                "min_speakers": 1,
                "huggingface_access_token": os.getenv("HUGGINGFACE_TOKEN"),
                "language_detection_min_prob": 0,
                "language_detection_max_tries": 5,
            },
        )

        while prediction.status not in ["succeeded", "failed"]:
            logging.info(
                f"Transcription status: {prediction.status}. Waiting for completion..."
            )
            time.sleep(5)
            prediction.reload()

        if prediction.status == "succeeded":
            pred_transcript = prediction.output
        if prediction.status == "failed":
            raise ModelError(
                prediction=prediction,
                message="Transcription failed",
                logs=prediction.logs,
            )

        if not pred_transcript:
            raise Exception(f"Transcription failed. No transcript returned.")

        self._save_queue.append(
            Transcript(
                podcast_episode_url=self.url, properties=pred_transcript
            )
        )

        self.save()

    def process_transcription(
        self, overwrite: bool = False, overwrite_if_stale: bool = False
    ):
        """
        Post-process transcription to combine speaker turns into paragraphs,
        and clean up the text using a language model.
        This will also add summaries to each paragraph.
        """

        if self._properties.get("paragraphs") and not overwrite:
            logging.info("Paragraphs already exist. Skipping!")
            return

        transcript = Transcript(podcast_episode_url=self.url).load()
        if not transcript:
            raise Exception("No transcript to post-process.")

        cleaner = dspy.ChainOfThought(CleanTranscript)

        paragraphs = []
        for turn in self._combine_turns(transcript._properties):
            pred = cleaner(
                speech=turn["speech"],
                context=f"Episode description: {self._properties.get('description', '')}",
            )

            turn["text"] = pred.text
            turn["one_sentence"] = pred.one_sentence
            turn["few_sentences"] = pred.few_sentences

            print("\n----------------- start of turn -----------------")
            print(f"# speaker: {turn['speaker']}")
            print(f"---\n# speech: {turn["speech"]}")
            print(f"---\n# text: {pred.text}")
            print(f"---\n# one_sentence: {pred.one_sentence}")
            print(f"---\n# few_sentences: {pred.few_sentences}")
            print("=================================================")

            paragraphs.append(turn)

        self._properties["paragraphs"] = paragraphs

        self.save()

    def _combine_turns(self, transcript: dict) -> list[dict]:
        """Combine speaker turns into one full turn."""

        logging.info("Combining speaker turns into paragraphs...")

        turns = transcript.get("segments", [])

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
            if (
                prev_turn.get("speaker") == turn.get("speaker")
                and len(turn.get("speaker", "")) > 0
            ):
                prev_turn = turn
                current_texts.append(turn.get("text"))
                continue

            # new speaker, record combined turn
            else:

                # store combined turn
                combined_turns.append(
                    {
                        "end": prev_turn.get("end"),
                        "speaker": prev_turn.get("speaker"),
                        "start": start_of_combined_turn,
                        "speech": " ".join(current_texts),
                    }
                )

                # mark new start
                start_of_combined_turn = turn.get("start")
                current_texts = [turn.get("text")]
                prev_turn = turn

        # last turn
        combined_turns.append(
            {
                "end": prev_turn.get("end"),
                "speaker": prev_turn.get("speaker"),
                "start": start_of_combined_turn,
                "speech": " ".join(current_texts),
            }
        )

        return combined_turns

    def define_speakers(self, overwrite: bool = False):
        """
        Define Person nodes (host, guests) from the transcript.
        This will create Person nodes and connect them to the PodcastEpisode.
        """
        if not self._properties.get("paragraphs"):
            raise Exception("No paragraphs to define speakers from.")

        if self._properties.get("speakers") and not overwrite:
            logging.info("Speakers already defined. Skipping.")
            self._ensure_speakers_in_paragraphs()
            return

        logging.info(
            f"Defining speakers from the transcript... overwrite={overwrite}"
        )

        context = f"""
        Podcast episode title: {self._properties.get("title", "Unknown Title")}
        Podcast episode description: {self._properties.get("description", "No description provided.")}
        First few paragraphs of the transcript:
        {self._properties.get("paragraphs", {})[:2]}  # Show only first 3 paragraphs for context
        """

        speaker_definer = dspy.ChainOfThought(DefineSpeakers)
        pred = speaker_definer(context=context)

        if not pred.speakers:
            raise Exception("No speakers defined in the transcript.")

        logging.info(f"Defined speakers: {pred.speakers}")

        self._properties["speakers"] = pred.speakers

        for speaker in self._properties.get("speakers", []):
            # Create Person node and set episode relationship
            person = Person(
                properties=speaker,
                podcast_episode=self,
                role=speaker.get("role", "guest").lower(),
            )
            self._save_queue.append(person)

        # Ensure speakers are added to paragraphs
        self._ensure_speakers_in_paragraphs()

        self.save()

    def _ensure_speakers_in_paragraphs(self):
        """
        Ensure that speakers are added to paragraphs.
        """
        speakers = self._properties.get("speakers", [])
        for para in self._properties.get("paragraphs", []):
            speaker_id = para.get("speaker")
            if speaker_id:
                # Find the speaker in the defined speakers
                speaker = next(
                    (s for s in speakers if s["speaker_id"] == speaker_id),
                    None,
                )
                if speaker:
                    para["speaker"] = speaker

    def paragraphize(self, overwrite: bool = False):
        """
        Create Paragraph nodes from the transcript.
        Each paragraph is a single speaker turn, combined if necessary.
        """
        paragraphs = self._properties.get("paragraphs")

        if not paragraphs:
            raise Exception("No paragraphs to create from.")

        logging.info(
            f"Creating Paragraph nodes for {len(paragraphs)} paragraphs..."
        )

        # clean up paragraphs, remove empty turns
        logging.info(
            f"Cleaning up paragraphs... {len(paragraphs)} paragraphs before cleanup."
        )
        paragraphs = [
            para
            for para in paragraphs
            if len(para.get("speech")) and type(para.get("speaker")) is dict
        ]
        logging.info(
            f"Cleaning up paragraphs... {len(paragraphs)} paragraphs after cleanup."
        )

        if not paragraphs:
            raise Exception("No paragraphs to create from.")

        previous_paragraph = None
        for para in paragraphs:
            paragraph = Paragraph(
                properties=para,
                podcast_episode=self,
                previous_paragraph=previous_paragraph,
            )
            self._save_queue.append(paragraph)
            previous_paragraph = paragraph

        self.save()

    def post_download_hook(self, info):
        """Post-download hook for yt-dlp to store metadata after download."""
        if info.get("status") == "finished":
            self._properties.update(
                {
                    "filename": info["info_dict"]["_filename"],
                    "duration": info["info_dict"]["duration"],
                    "date": datetime.strptime(
                        info["info_dict"]["upload_date"], "%Y%m%d"
                    ).date(),
                    "description": info["info_dict"]["description"],
                    "display_id": info["info_dict"]["display_id"],
                }
            )

    def _get_filename(self):
        """Extract filename from the full path stored in post_download_hook"""
        filename = self._properties.get("filename")
        return os.path.basename(filename)


class Person:
    """Person node in the graph, representing a speaker."""

    def __init__(
        self,
        properties: dict,
        podcast_episode: PodcastEpisode = None,
        role: str = None,
    ):
        self._properties = properties
        self.podcast_episode = podcast_episode
        self.role = role

    def save(self):
        """Save Person node to Memgraph and create relationships."""

        # Build SET clause dynamically
        filtered_properties = {
            k: v
            for k, v in self._properties.items()
            if k not in ["role", "speaker_id"]
        }
        set_clause = ", ".join(
            [f"p.{key} = ${key}" for key in filtered_properties.keys()]
            + ["p.updated_at = datetime()"]
        )

        query = f"""
        MERGE (p:Person {{full_name: $full_name}})
        SET {set_clause}
        RETURN p
        """

        records, summary, keys = memgraph_query(
            query,
            **self._properties,
        )

        logging.info(
            f"Saved Person node: {self._properties.get('full_name', 'Unknown')}"
        )

        # Create relationship based on role
        if self.podcast_episode and self.role:
            relationship_type = (
                "IS_HOST_OF" if self.role == "host" else "IS_GUEST_ON"
            )

            records, summary, keys = memgraph_query(
                f"""
                MATCH (p:Person {{full_name: $full_name}})
                MATCH (pe:PodcastEpisode {{url: $podcast_episode_url}})
                MERGE (p)-[:{relationship_type}]->(pe)
                """,
                full_name=self._properties["full_name"],
                podcast_episode_url=self.podcast_episode.url,
            )

            logging.info(
                f"Created {relationship_type} relationship for {self._properties['full_name']}"
            )


class Paragraph:
    """Each paragraph of text, from a single speaker."""

    def __init__(
        self,
        properties: dict,
        podcast_episode: PodcastEpisode,
        previous_paragraph: "Paragraph" = None,
    ):
        self._properties = properties
        self._podcast_episode = podcast_episode
        self._previous_paragraph = previous_paragraph

    def save(self):
        """Save Paragraph node to Memgraph and create relationships."""

        # Build SET clause dynamically
        set_clause = ", ".join(
            [f"p.{key} = ${key}" for key in self._properties.keys()]
            + ["p.updated_at = datetime()"]
        )

        query = f"""
        MERGE (p:Paragraph {{start: $start, end: $end, podcast_episode_url: $podcast_episode_url}})
        SET {set_clause}
        RETURN p
        """

        records, summary, keys = memgraph_query(
            query,
            **self._properties,
            podcast_episode_url=self._podcast_episode.url,
        )

        logging.debug(
            f"Saved Paragraph node with start: {self._properties['start']}"
        )

        if not self._previous_paragraph:
            # Create relationship to PodcastEpisode
            memgraph_query(
                """
                MATCH (pe:PodcastEpisode {url: $podcast_episode_url})
                MATCH (p:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
                MERGE (pe)-[:FIRST_PARAGRAPH]->(p)
                """,
                start=self._properties["start"],
                end=self._properties["end"],
                podcast_episode_url=self._podcast_episode.url,
            )
            logging.debug(
                f"Created FIRST_PARAGRAPH relationship for {self._properties['start']}"
            )

        if self._previous_paragraph:
            # Create relationship to previous Paragraph
            memgraph_query(
                """
                MATCH (p1:Paragraph {start: $prev_start, end: $prev_end, podcast_episode_url: $podcast_episode_url})
                MATCH (p2:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
            MERGE (p1)-[:NEXT_PARAGRAPH]->(p2)
            """,
                start=self._properties["start"],
                end=self._properties["end"],
                prev_start=self._previous_paragraph._properties["start"],
                prev_end=self._previous_paragraph._properties["end"],
                podcast_episode_url=self._podcast_episode.url,
            )
            logging.debug(
                f"Created NEXT_PARAGRAPH relationship for {self._properties['start']}"
            )

        # Create relationship to Person (SPOKEN_BY)
        speaker_dict = self._properties.get("speaker")
        if speaker_dict:
            full_name = speaker_dict.get("full_name")
            if full_name:
                memgraph_query(
                    """
                    MATCH (p:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
                    MATCH (person:Person {full_name: $full_name})
                    MERGE (p)-[:EXPRESSED_BY]->(person)
                    """,
                    start=self._properties["start"],
                    end=self._properties["end"],
                    podcast_episode_url=self._podcast_episode.url,
                    full_name=full_name,
                )
                logging.debug(
                    f"Created EXPRESSED_BY relationship for {self._properties['start']} with speaker {full_name}"
                )


class Sentence:
    """Could be cool, but too complicated for now."""

    def __init__(self, text: str):
        pass


class Pipeline:
    def __init__(
        self, podcast: Podcast, max_episodes: int = 3, max_workers: int = 5
    ):
        self.podcast = podcast
        self.max_episodes = max_episodes
        self.max_workers = max_workers

    def run_podcast(self):

        # TODO:
        # - √ clean up PodcastEpisode properties (too heavy)
        # - √ Add Transcript node
        # - create embeddings of all text
        #   > on paragraphs (text, one_sentence, few_sentences)
        #   > on person (full_name + description)
        #   > on podcast_episode (title + description)
        # - add stats (word count, sentence count)
        # - enrich Person nodes with DBPedia URIs
        # - ENTITIES
        # - add audio clips (later)
        # FIXME: bugs
        # - when two or more speakers, issue with parsing
        # - Person node without name should not be created

        episodes = self.podcast.PodcastEpisodes
        if not episodes:
            episodes = self.podcast.load_episodes()

        # Filter out episodes that are already processed
        logging.info(
            f"Filtering episodes that are already processed... Total: {len(episodes)}"
        )
        episodes = [
            episode
            for episode in episodes
            if not episode._properties.get("updated_at")
        ]
        logging.info(f"Filtered episodes: {len(episodes)}")

        logging.info(
            f"Pipeline episodes: {len(episodes)}. Handling {self.max_episodes} episodes."
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self.run_episode_tasks, episode)
                for episode in episodes[: self.max_episodes]
            ]

        concurrent.futures.wait(futures)

    def run_episode_tasks(self, episode: PodcastEpisode):
        logging.info("")
        logging.info(f">>>>>>>>>>>>>>")
        logging.info(f">> Pipeline: Running episode: {episode.url}")
        logging.info(f">>>>>>>>>>>>>>")

        try:

            episode.load()  # Load existing episode properties if available

            # download
            episode.download()

            # transcribe
            episode.transcribe()

            # clean up transcription / diarization
            episode.process_transcription(overwrite=True)

            # define Person nodes (host, guests)
            episode.define_speakers(overwrite=True)

            # create Paragraph nodes from transcription
            episode.paragraphize(overwrite=True)

            # create embeddings for all text
            # episode.create_embeddings(overwrite=True)

            # create statistics

        except Exception as e:
            logging.exception(f"Error processing episode {episode.url}: {e}")
            logging.warning("Episode was not saved to graph.")


def init_memgraph():
    """
    Initialize Memgraph database with constraints and indexes.
    """
    with memgraph.session() as session:
        # Create constraint for Podcast
        session.run("CREATE CONSTRAINT ON (p:Podcast) ASSERT p.url IS UNIQUE")
        logging.info("Created constraint for Podcast.")

        # Create constraint for PodcastEpisode
        session.run(
            "CREATE CONSTRAINT ON (pe:PodcastEpisode) ASSERT pe.url IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT ON (pe:PodcastEpisode) ASSERT pe.url IS UNIQUE"
        )
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


def main():
    # init_memgraph()
    # clear_memgraph()

    podcast = Podcast(url="https://www.youtube.com/@GDiesen1/videos")
    podcast.fetch_meta()
    podcast.save()
    podcast.load()
    episodes = podcast.load_episodes()
    logging.info(f"Podcast episodes: {len(episodes)}")

    pipeline = Pipeline(podcast, max_episodes=300, max_workers=20)
    pipeline.run_podcast()


if __name__ == "__main__":
    main()
