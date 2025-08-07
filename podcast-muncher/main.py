from functools import wraps
from yt_dlp import YoutubeDL
import replicate
from replicate.exceptions import ReplicateError
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
import secrets
import string


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

lm = dspy.LM(
    "openai/gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=32768,
)
dspy.configure(lm=lm)


def random_string(length: int = 10) -> str:
    """Generate a random string of fixed length."""
    letters = string.ascii_letters + string.digits
    return "".join(secrets.choice(letters) for _ in range(length))


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
            |>> Paragraph<-SAID-Person

        - Person
            |name
            |dbpedia_uri
            |profession
            |
            |>> Paragraph<-SAID-Person


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

    def get_episode(self, url: str) -> "PodcastEpisode":
        """
        Get a PodcastEpisode by its URL.
        If it doesn't exist, return None.
        """
        for episode in self.PodcastEpisodes:
            if episode.url == url:
                return episode
        return None


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
        MERGE (pe)-[ht:HAS_TRANSCRIPT]->(t)
        SET ht.updated_at = datetime()
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
        MERGE (p)-[he:HAS_EPISODE]->(pe)
        SET he.updated_at = datetime()
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

    def download(self, rerun: bool = False):
        """Download episode using yt-dlp. Save to file."""

        transcript = Transcript(podcast_episode_url=self.url).load()
        if transcript and not rerun:
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

    def transcribe(self, rerun: bool = False):
        """
        Transcribe using Replicate interference.
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for model details
        """

        transcript = Transcript(podcast_episode_url=self.url).load()
        if transcript and not rerun:
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
                "language": "en",
                "language_detection_min_prob": 0.8,
                "language_detection_max_tries": 3,
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
            raise ReplicateError(
                prediction=prediction,
                title="Transcription failed",
                detail=prediction.logs,
            )

        if not pred_transcript:
            raise Exception(f"Transcription failed. No transcript returned.")

        self._save_queue.append(
            Transcript(
                podcast_episode_url=self.url, properties=pred_transcript
            )
        )

        self.save()

    def summarize_transcript(self, rerun: bool = False):
        """
        Post-process transcription to combine speaker turns into paragraphs,
        and clean up the text using a language model.
        This will also add summaries to each paragraph.
        """

        if self._properties.get("paragraphs") and not rerun:
            logging.info("Paragraphs already exist. Skipping!")
            return

        transcript = Transcript(podcast_episode_url=self.url).load()

        if not transcript:
            raise Exception("No transcript to post-process.")

        cleaner = dspy.ChainOfThought(CleanTranscript)

        paragraphs = []
        turns = self._combine_turns(transcript._properties)

        logging.info(f"Combined speaker turns into {len(turns)} paragraphs.")

        for turn in turns:
            pred = cleaner(
                speech=turn["speech"],
                context=f"Episode description: {self._properties.get('description', '')}",
            )

            turn["text"] = pred.text
            turn["one_sentence"] = pred.one_sentence
            turn["few_sentences"] = pred.few_sentences

            print("\n----------------- start of turn -----------------")
            print(f"\n-> speaker: \n{turn['speaker']}")
            print(f"\n-> speech: \n{turn["speech"]}")
            print(f"\n-> text: \n{pred.text}")
            print(f"\n-> one_sentence: \n{pred.one_sentence}")
            print(f"\n-> few_sentences: \n{pred.few_sentences}")
            print("==================== end ==========================")

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
                # filter out empty speaker names
                and len(turn.get("speaker", "")) > 0
                # Limit to 100 turns per paragraph
                and len(current_texts) < 100
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

    def define_speakers(self, rerun: bool = False):
        """
        Define Person nodes (host, guests) from the transcript.
        This will create Person nodes and connect them to the PodcastEpisode.
        This will not connect paragraphs to speakers; this is done later.
        """
        if not self._properties.get("paragraphs"):
            raise Exception("No paragraphs to define speakers from.")

        if self._properties.get("speakers") and not rerun:
            logging.info("Speakers already defined. Skipping.")
            return

        logging.info(f"Defining speakers from the transcript... rerun={rerun}")

        context = f"""
        Podcast title: {self.podcast._properties.get("title", "No title provided.")}
        Podcast description: {self.podcast._properties.get("description", "No description provided.")}
        --
        Podcast episode title: {self._properties.get("title", "Unknown Title")}
        Podcast episode description: {self._properties.get("description", "No description provided.")}
        --
        First few paragraphs of the transcript:
        {self._properties.get("paragraphs", {})[:5]}
        """

        # TODO: Add check for expected number of speakers based on PodcastEpisode properties
        #       then check if all speakers are identified in the transcript
        #       if not, then do another run, but with all paragraphs in context

        logging.info(f"Context for speaker definition: {context}")

        speaker_definer = dspy.ChainOfThought(DefineSpeakers)
        pred = speaker_definer(context=context)

        logging.info(f"Predicted speakers: {pred}")

        if not pred.speakers:
            raise Exception("No speakers defined in the transcript.")

        logging.info(f"Defined speakers: {pred.speakers}")

        self._properties["speakers"] = [
            speaker
            for speaker in pred.speakers
            if speaker.get("full_name") is not None
        ]

        logging.info(
            f"""Defined {len(self._properties['speakers'])} speakers from the transcript: 
            {self._properties['speakers']}"""
        )

        self.save()

    def create_speakers(self, rerun: bool = False):
        """Create Person nodes for each speaker defined in the transcript."""
        for speaker in self._properties.get("speakers", []):
            # Create Person node and set episode relationship
            person = Person(
                properties=speaker,
                podcast_episode=self,
                role=speaker.get("role", "guest").lower(),
            )
            self._save_queue.append(person)

        # Ensure speakers are added to paragraphs
        # self._ensure_speakers_in_paragraphs()

        self.save()

    # def _ensure_speakers_in_paragraphs(self):
    #     """
    #     Ensure that speakers are added to paragraphs.
    #     """
    #     speakers = self._properties.get("speakers", [])
    #     for para in self._properties.get("paragraphs", []):
    #         speaker_id = para.get("speaker")
    #         if speaker_id:
    #             # Find the speaker in the defined speakers
    #             speaker = next(
    #                 (s for s in speakers if s["speaker_id"] == speaker_id),
    #                 None,
    #             )
    #             if speaker:
    #                 para["speaker"] = speaker

    def paragraphize(self, rerun: bool = False):
        """
        Create Paragraph nodes from the transcript.
        Each paragraph is a single speaker turn, combined if necessary.
        """
        paragraphs = self._properties.get("paragraphs")
        speakers = self._properties.get("speakers")

        if not paragraphs:
            raise Exception("No paragraphs to create from.")

        for pg in paragraphs:
            speaker_id = pg.get("speaker")
            if type(speaker_id) == str and len(speaker_id) > 3:
                speaker = next(
                    (s for s in speakers if s["speaker_id"] == speaker_id),
                    None,
                )
                if speaker:
                    pg["speaker"] = speaker

        paragraphs = [
            para
            for para in paragraphs
            if len(para.get("speech"))
            # and type(para.get("speaker")) is dict
        ]

        if not paragraphs:
            raise Exception("No paragraphs to create from.")

        logging.info(
            f"Creating Paragraph nodes for {len(paragraphs)} paragraphs..."
        )

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

    def debug_clean_up_speakers(self):
        paragraphs = self._properties.get("paragraphs", [])
        for paragraph in paragraphs:
            speaker = paragraph.get("speaker")
            while speaker and type(speaker) == dict:
                if (
                    "speaker_id" in speaker
                    and type(speaker["speaker_id"]) == dict
                ):
                    speaker = speaker["speaker_id"]
                else:
                    break
            paragraph["speaker"] = speaker
        self.save()


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

        if not self._properties.get("full_name"):
            logging.warning("No full_name provided. Skipping save.")
            return

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
                MERGE (p)-[rt:{relationship_type}]->(pe)
                SET rt.updated_at = datetime()
                RETURN p, pe
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
                MERGE (pe)-[np:NEXT_PARAGRAPH]->(p)
                SET np.updated_at = datetime()
                """,
                start=self._properties["start"],
                end=self._properties["end"],
                podcast_episode_url=self._podcast_episode.url,
            )

        if self._previous_paragraph:
            # Create relationship to previous Paragraph
            memgraph_query(
                """
                MATCH (p1:Paragraph {start: $prev_start, end: $prev_end, podcast_episode_url: $podcast_episode_url})
                MATCH (p2:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
                MERGE (p1)-[np:NEXT_PARAGRAPH]->(p2)
                SET np.updated_at = datetime()
                """,
                start=self._properties["start"],
                end=self._properties["end"],
                prev_start=self._previous_paragraph._properties["start"],
                prev_end=self._previous_paragraph._properties["end"],
                podcast_episode_url=self._podcast_episode.url,
            )

        # Create relationship to Person (SAID)
        speaker_dict = self._properties.get("speaker")
        if speaker_dict:
            full_name = speaker_dict.get("full_name")
            if full_name:
                memgraph_query(
                    """
                    MATCH (p:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
                    MATCH (person:Person {full_name: $full_name})
                    MERGE (person)-[eb:SAID]->(p)
                    SET eb.updated_at = datetime()
                    """,
                    start=self._properties["start"],
                    end=self._properties["end"],
                    podcast_episode_url=self._podcast_episode.url,
                    full_name=full_name,
                )


class Sentence:
    """Could be cool, but too complicated for now."""

    def __init__(self, text: str):
        pass


def debug_clean_up_speakers_in_paragraph_nodes():
    logging.info("debug_clean_up_speakers_in_paragraph_nodes")

    query = """
    MATCH (pg:Paragraph)
    return pg
    """
    records, summary, keys = memgraph_query(query)

    for record in records:
        paragraph = record["pg"]

        speaker = paragraph._properties["speaker"]
        while speaker and type(speaker) == dict:
            if "speaker_id" in speaker and type(speaker["speaker_id"]) == dict:
                speaker = speaker["speaker_id"]
            else:
                break

        query = """
        MATCH (pg:Paragraph)
        WHERE id(pg) = $element_id
        SET pg.speaker = $speaker
        """
        records, summary, keys = memgraph_query(
            query, element_id=paragraph.id, speaker=speaker
        )


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
        # - cleanup paragraphs from PodcatEpisode properties

        # FIXME: bugs
        # - √ when three or more speakers, issue with parsing
        #   - how to solve?
        #     a) use short video clip?
        #     b) match audio with known audio of speakers?
        #     c) look through whole transcript for hint to speaker name per turn?
        #     ... tricky. need backup, ie. add paragraphs even without speakers
        # - √ when only one speaker, issue with parsing (also will have too long paragraph)
        # - √ Person node without name should not be created
        # - √ clean up messy edges

        episodes = self.podcast.PodcastEpisodes
        if not episodes:
            episodes = self.podcast.load_episodes()

        # Filter out episodes that are already processed
        # logging.info(
        #     f"Filtering episodes that are already processed... Total: {len(episodes)}"
        # )
        # episodes = [
        #     episode
        #     for episode in episodes
        #     if not episode._properties.get("updated_at")
        # ]
        # logging.info(f"Filtered episodes: {len(episodes)}")

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
            episode.summarize_transcript()

            # define speakers
            episode.define_speakers(rerun=True)

            # create Person nodes for speakers
            episode.create_speakers(rerun=True)

            # create Paragraph nodes from transcription
            episode.paragraphize(rerun=True)

            # create embeddings for all text
            # episode.create_embeddings(rerun=True)

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
        session.run(
            "CREATE CONSTRAINT ON (pe:PodcastEpisode) ASSERT pe.url IS UNIQUE"
        )
        session.run("CREATE INDEX ON :Person")
        session.run("CREATE INDEX ON :Person(full_name)")
        session.run("CREATE INDEX ON :Transcript")
        session.run("CREATE INDEX ON :Transcript(podcast_episode_url)")
        session.run("CREATE INDEX ON :Paragraph")
        session.run("CREATE INDEX ON :Paragraph(podcast_episode_url)")
        session.run("CREATE INDEX ON :Paragraph(start)")
        session.run("CREATE INDEX ON :Paragraph(end)")
        session.run("CREATE INDEX ON :PodcastEpisode")
        session.run("CREATE INDEX ON :PodcastEpisode(url)")


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

    # debug_clean_up_speakers_in_paragraph_nodes()

    podcast = Podcast(url="https://www.youtube.com/@GDiesen1/videos")
    podcast.fetch_meta()
    podcast.save()
    podcast.load()
    episodes = podcast.load_episodes()
    logging.info(f"Podcast episodes: {len(episodes)}")

    # for episode in episodes:
    # episode.debug_clean_up_speakers()

    pipeline = Pipeline(podcast, max_episodes=1, max_workers=1)
    pipeline.run_podcast()

    # episode = podcast.get_episode(
    #     url="https://www.youtube.com/watch?v=SWux-RBbKGs"
    # )
    # if episode:
    #     pipeline.run_episode_tasks(episode)


if __name__ == "__main__":
    main()
