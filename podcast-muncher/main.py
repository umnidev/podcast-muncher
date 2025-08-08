from functools import wraps
import voyageai
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
from llm import (
    CleanTranscript,
    DefineSpeakers,
    ExtractEntities,
    AssignDBPediaUri,
    DetermineDuplicateToKeep,
)
import concurrent.futures
import secrets
import string
from typing import Dict, List, Tuple


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

NER_VERSION = f"{os.getenv("NER_VERSION", "v0.2.1")}-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}"
logging.info(f"Using NER version: {NER_VERSION}")

fast_lm = dspy.LM(
    # "openai/gpt-4.1-2025-04-14",
    model=os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini-2025-08-07"),
    api_key=os.getenv("OPENAI_API_KEY"),
    max_completion_tokens=100000,
    max_tokens=None,
    temperature=1,
)
strong_lm = dspy.LM(
    model=os.getenv("OPENAI_STRONG_MODEL", "gpt-5-2025-08-07"),
    api_key=os.getenv("OPENAI_API_KEY"),
    max_completion_tokens=100000,
    max_tokens=None,
    temperature=1,
)

# default model
dspy.configure(lm=fast_lm)


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
    logging.debug(f"Executing query: {query} with params: {params}")
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
            logging.debug("No meta, can't load episodes.")
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

    def save(self):
        if not self._properties:
            logging.debug("No properties to save.")
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
            logging.debug("Already downloaded. Skipping!")
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

    def transcribe(self, rerun: bool = False):
        """
        Transcribe using Replicate interference.
        See https://replicate.com/predictions?prediction=8e1g02sdj1rj60crc8ar2kpz4m for model details
        """

        transcript = Transcript(podcast_episode_url=self.url).load()
        if transcript and not rerun:
            logging.debug("Transcript already exists. Skipping!")
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
            logging.info(
                f"Transcription succeeded. Transcript length: {len(pred_transcript.get('segments', []))} segments."
            )
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
            logging.debug("Paragraphs already exist. Skipping!")
            return

        transcript = Transcript(podcast_episode_url=self.url).load()

        if not transcript:
            raise Exception("No transcript to post-process.")

        cleaner = dspy.ChainOfThought(CleanTranscript)

        paragraphs = []
        turns = self._combine_turns(transcript._properties)

        logging.debug(f"Combined speaker turns into {len(turns)} paragraphs.")

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

        logging.debug("Combining speaker turns into paragraphs...")

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
            logging.debug("Speakers already defined. Skipping.")
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

        logging.debug(f"Context for speaker definition: {context}")

        speaker_definer = dspy.ChainOfThought(DefineSpeakers)
        pred = speaker_definer(context=context)

        logging.debug(f"Predicted speakers: {pred}")

        if not pred.speakers:
            raise Exception("No speakers defined in the transcript.")

        logging.debug(f"Defined speakers: {pred.speakers}")

        self._properties["speakers"] = [
            speaker
            for speaker in pred.speakers
            if speaker.get("name") is not None
        ]

        logging.debug(
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

        self.save()

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
            para for para in paragraphs if len(para.get("speech")) > 0
        ]

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

    def extract_entities(self, rerun: bool = False):
        logging.info(
            f"Extracting entities for episode {self._properties.get("title")}"
        )

        # get paragraphs
        query = """
        MATCH (p:Paragraph)
        WHERE p.podcast_episode_url = $podcast_episode_url
        RETURN p
        """
        records, summary, keys = memgraph_query(
            query, podcast_episode_url=self.url
        )

        for record in records:
            properties = record["p"]._properties
            paragraph = Paragraph(properties=properties, podcast_episode=self)
            paragraph.extract_entities(rerun=rerun)
            paragraph.save()

    def create_stats(self):
        # word count per episode
        # word count per paragraph

        pass

    def create_embeddings(self, rerun: bool = False):
        """Create vector embeddings for nodes and edges."""
        # paragraph text
        # paragraph few_sentence (?)
        # edge description
        #
        # just have to test it out:
        # - edge.descriptions (small) linked to source_paragraph_id
        #   - add all 20 edges as separate chunks, and append whole paragraph as last chunk for more context
        # - paragraph as chunk in list of all paragraphs in PodcastEpisode
        # - few_sentences as chunk in list of all few_sentences in (pe)
        # the POINT when running a query, is to find:
        # - FACTS (edges)
        # - the relevant paragraphs (p:Paragraph nodes)
        #

        # get paragraphs from graph
        query = """
        MATCH (p:Paragraph)
        WHERE p.podcast_episode_url = $podcast_episode_url
        RETURN p
        """
        records, summary, keys = memgraph_query(
            query, podcast_episode_url=self.url
        )
        if not records or len(records) == 0:
            logging.warning(
                f"No paragraphs found for podcast episode {self.url}. Skipping embedding generation."
            )
            return
        # create list of paragraphs to embed
        paragraphs = []
        for record in records:
            properties = record["p"]._properties
            text = properties.get("text", "")
            if not text or len(text) < 200:
                continue

            speaker = properties.get("speaker", {}).get(
                "name", "Unknown Speaker"
            )

            p = {
                "id": record["p"].element_id,
                "speaker": speaker,
                "text": text,
                "few_sentences": properties.get("few_sentences", ""),
                "one_sentence": properties.get("one_sentence", ""),
                "paragraph_id": properties.get("paragraph_id", ""),
            }

            paragraphs.append(p)

        text_inputs = []
        for p in paragraphs:
            # create text input for embedding
            text_input = f"{p['speaker']} said: \n {p['text']}"
            text_inputs.append(text_input)

        embeddings = self._generate_document_embedding([text_inputs])

        # Debug: Check available attributes
        print(f"Available attributes: {dir(embeddings)}")
        print(f"Type: {type(embeddings)}")

        # Try different possible attribute names
        for attr in ["data", "embedding", "embeddings", "results", "output"]:
            if hasattr(embeddings, attr):
                print(f"Found attribute '{attr}': {getattr(embeddings, attr)}")

        # Access the results and extract embeddings
        embedding_results = embeddings.results
        print(f"Number of embedding results: {len(embedding_results)}")

        # Extract embeddings and indices for later use
        embeddings_list = []
        indices_list = []

        result = embedding_results[0] if embedding_results else None
        if not result:
            logging.error("No embedding results found. Exiting.")
            return

        # for result in embedding_results:
        #     indices_list.append(result.index)
        #     embeddings_list.append(result.embeddings)
        #     print(
        #         f"Index: {result.index}, Embedding shape: {len(result.embeddings) if result.embeddings else 0}"
        #     )

        print(f"Total tokens used: {embeddings.total_tokens}")
        print(f"Total paragraphs: {len(paragraphs)}")

        for i, p in enumerate(paragraphs):
            p["em_text"] = result.embeddings[i]

        # save embeddings to graph
        for p in paragraphs:
            print(f"Saving embedding for paragraph {p}...")
            if "em_text" in p:
                query = """
                MATCH (p:Paragraph)
                WHERE id(p) = $id
                SET p.em_text = $em_text
                RETURN p
                """
                records, summary, keys = memgraph_query(
                    query, id=int(p["id"]), em_text=p["em_text"]
                )
                if records and len(records):
                    logging.info(f"Saved embedding for paragraph {p['id']}.")
                else:
                    logging.warning(
                        f"Failed to save embedding for paragraph {p['id']}."
                    )

    def _generate_document_embedding(self, inputs: List[List[str]]):
        """Generate document embeddings using VoyageAI."""
        logging.info(
            f"Generating document embeddings for {len(inputs)} inputs."
        )
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        embeddings = vo.contextualized_embed(
            inputs=inputs,
            model="voyage-context-3",
            output_dimension=512,
            input_type="document",
        )
        # ContextualizedEmbeddingsObject
        logging.info(
            f"Generated embeddings with {embeddings.total_tokens} tokens."
        )
        # logging.info(f"Embedding results: {embeddings.results}")
        return embeddings


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

    def extract_entities(self, rerun: bool = False):
        text = self._properties["text"]

        if not text or not len(text):
            return logging.info("No text to extract entities from. Skipping.")

        if self._properties.get("entities") and not rerun:
            logging.info("Entities already extracted. Skipping.")
            return

        # extract entities using dspy
        with dspy.context(lm=strong_lm):
            logging.info(
                f"Extracting entities from paragraph with strong model."
            )
            extractor = dspy.ChainOfThought(ExtractEntities)
            pred = extractor(text=text)

        self._debug_log_pred(pred)

        # create extracted entities and relations in the graph
        ner_ids = self._create_entities(pred)
        self._create_relations(pred, ner_ids)

    def _debug_log_pred(self, pred):
        """Log the prediction details for debugging."""

        if len(pred.entities):
            for entity in pred.entities:
                logging.info(
                    f"Entity --> {entity.get('name')} ({entity['type']} (DBPedia: {entity.get('dbpedia_uri')})) "
                )

        if len(pred.relations):
            for edge in pred.relations:
                source = next(
                    (
                        ent
                        for ent in pred.entities
                        if ent["ner_id"] == edge["source"]
                    ),
                    None,
                )
                target = next(
                    (
                        ent
                        for ent in pred.entities
                        if ent["ner_id"] == edge["target"]
                    ),
                    None,
                )
                if source and target:
                    logging.info(
                        f"""Relation --> {source["name"]} {edge["name"]} {target["name"]} (DBPedia: {edge.get("dbo_type", "")})"""
                    )

    def _create_entities(self, pred):
        """Find existing entities in the graph or create new ones based on the prediction."""
        ner_ids = {}

        for entity in pred.entities:

            if entity.get("confidence", 0) < 0.8:
                logging.debug(f"Confidence too low, skipping. {entity}")
                continue

            label = (
                entity["dbo_type"].split(":")[1]
                if entity["dbo_type"]
                else entity["type"].title().replace(" ", "")
            )
            dbpedia_uri = entity.get("dbpedia_uri")
            ner_id = entity.get("ner_id")
            params = {
                k: v
                for k, v in entity.items()
                if k not in ["ner_id", "confidence", "details"]
            }
            params["dbpedia_uri"] = dbpedia_uri
            params["ner_version"] = NER_VERSION

            existing = self._find_existing_entity(
                label, dbpedia_uri=dbpedia_uri, name=params.get("name")
            )

            set_clause = ", ".join(
                [f"n.{key} = ${key}" for key in params.keys()]
                + ["n.updated_at = datetime()"]
            )
            if not existing:
                query = f"""
                CREATE (n:{label})
                SET {set_clause}
                RETURN n
                """
                records, summary, keys = memgraph_query(query, **params)

            else:
                query = f"""
                MATCH (n)
                WHERE id(n) = $id
                SET {set_clause}
                RETURN n
                """
                records, summary, keys = memgraph_query(
                    query, id=int(existing.element_id), **params
                )

            # store ner_id for later use
            ner_ids[ner_id] = records[0]["n"].element_id

        self._properties["entities"] = pred.entities

        return ner_ids

    def _create_relations(self, pred, ner_ids):
        """Create relations between entities based on the prediction."""
        if not pred.relations:
            logging.debug("No relations to create.")
            return

        for edge in pred.relations:
            if edge.get("confidence", 0) < 0.8:
                logging.debug(f"Confidence too low, skipping. {edge}")
                continue

            source_id = ner_ids.get(edge["source"])
            target_id = ner_ids.get(edge["target"])

            if not source_id or not target_id:
                logging.debug(f"Source or target not found for edge: {edge}")
                continue

            # get paragraph node
            source_paragraph = self._get_node()
            if not source_paragraph:
                logging.error(f"Paragraph node not found. Skipping. ")
                continue

            query = f"""
            MATCH (s), (t)
            WHERE id(s) = $source_id AND id(t) = $target_id
            MERGE (s)-[r:`{edge.get("name")}`]->(t)
            SET 
                r.updated_at = datetime(), 
                r.confidence = $confidence, 
                r.ner_version = $ner_version,
                r.description = $description,
                r.dbo_type = $dbo_type,
                r.source_paragraph_id = $source_paragraph_id
            RETURN s, t, r
            """
            records, summary, keys = memgraph_query(
                query,
                source_id=int(source_id),
                target_id=int(target_id),
                confidence=float(edge.get("confidence")),
                ner_version=NER_VERSION,
                description=edge.get("description", ""),
                dbo_type=edge.get("dbo_type", ""),
                source_paragraph_id=int(source_paragraph.element_id),
            )

        # store entities and relations in properties
        self._properties["relations"] = pred.relations

    def _get_node(self):
        query = """
        MATCH (p:Paragraph)
        WHERE p.podcast_episode_url = $podcast_episode_url
            AND p.start = $start 
            AND p.end = $end
        RETURN p
        """
        records, summary, keys = memgraph_query(
            query,
            podcast_episode_url=self._podcast_episode.url,
            start=self._properties["start"],
            end=self._properties["end"],
        )
        return records[0]["p"] if records and len(records) > 0 else None

    def _find_existing_entity(self, label, dbpedia_uri=None, name=None):
        """Try to find an existing entity by dbpedia_uri or name."""
        if dbpedia_uri:
            logging.info(
                f"Trying to find existing entity by DBpedia URI: {dbpedia_uri}"
            )
            # logging.warning(
            #     "OVERRIDE: ENCOURAGE DUPLICATE ENTITIES. RETURNING!"
            # )
            # return None  # TODO: remove this override

            query = f"""
                MATCH (n:{label})
                WHERE n.dbpedia_uri = $dbpedia_uri
                RETURN n
            """
            logging.info(f"Query: {query} with dbpedia_uri={dbpedia_uri}")

            records, summary, keys = memgraph_query(
                query, dbpedia_uri=dbpedia_uri
            )
            logging.info(f"Records found: {len(records)}")
            if records:
                return records[0]["n"]

        if name:
            query = f"""
                MATCH (n:{label})
                WHERE n.name = $name
                RETURN n
            """
            records, summary, keys = memgraph_query(query, name=f"`{name}`")
            if records:
                return records[0]["n"]

        return None

    def save(self):
        """Save Paragraph node to Memgraph and create relationships."""

        # Build SET clause dynamically
        set_clause = ", ".join(
            [
                f"p.{key} = ${key}"
                for key in self._properties.keys()
                if key != "podcast_episode_url"
            ]
            + ["p.updated_at = datetime()"]
        )
        properties = {
            k: v
            for k, v in self._properties.items()
            if k != "podcast_episode_url"
        }

        query = f"""
        MERGE (p:Paragraph {{start: $start, end: $end, podcast_episode_url: $podcast_episode_url}})
        SET {set_clause}
        RETURN p
        """

        records, summary, keys = memgraph_query(
            query,
            podcast_episode_url=self._podcast_episode.url,
            **properties,
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
            name = speaker_dict.get("name")
            if name:
                memgraph_query(
                    """
                    MATCH (p:Paragraph {start: $start, end: $end, podcast_episode_url: $podcast_episode_url})
                    MATCH (person:Person {name: $name})
                    MERGE (person)-[eb:SAID]->(p)
                    SET eb.updated_at = datetime()
                    """,
                    start=self._properties["start"],
                    end=self._properties["end"],
                    podcast_episode_url=self._podcast_episode.url,
                    name=name,
                )


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

        if not self._properties.get("name"):
            logging.warning("No name provided. Skipping save.")
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
        MERGE (p:Person {{name: $name}})
        SET {set_clause}
        RETURN p
        """

        records, summary, keys = memgraph_query(
            query,
            **self._properties,
        )

        logging.debug(
            f"Saved Person node: {self._properties.get('name', 'Unknown')}"
        )

        # Create relationship based on role
        if self.podcast_episode and self.role:
            relationship_type = (
                "IS_HOST_OF" if self.role == "host" else "IS_GUEST_ON"
            )

            records, summary, keys = memgraph_query(
                f"""
                MATCH (p:Person {{name: $name}})
                MATCH (pe:PodcastEpisode {{url: $podcast_episode_url}})
                MERGE (p)-[rt:{relationship_type}]->(pe)
                SET rt.updated_at = datetime()
                RETURN p, pe
                """,
                name=self._properties["name"],
                podcast_episode_url=self.podcast_episode.url,
            )

            logging.debug(
                f"Created {relationship_type} relationship for {self._properties['name']}"
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


def add_dbpedia_uris_to_persons():

    query = """
    MATCH (p:Person)
    RETURN p
    """
    records, summary, keys = memgraph_query(query)

    assigner = dspy.ChainOfThought(AssignDBPediaUri)

    for record in records:
        person = record["p"]._properties
        context = f"""
        Name: {person.get("name", None)}
        Title: {person.get("title", None)}
        Description: {person.get("description_long", person.get("description", None))}
        """
        pred = assigner(context=context)

        logging.debug(f"Got pred: {pred}")
        dbpedia_uri = pred.dbpedia_uri

        if dbpedia_uri == "None":
            logging.debug("No DBPedia")
            continue

        query = """
        MATCH (p:Person)
        WHERE id(p) = $element_id
        SET p.dbpedia_uri = $dbpedia_uri
        """
        records, summary, keys = memgraph_query(
            query, element_id=record["p"].id, dbpedia_uri=dbpedia_uri
        )

        logging.debug(f"Updated dbpedia_uri: {dbpedia_uri}")


class Pipeline:
    def __init__(
        self,
        podcast: Podcast,
        max_episodes: int = 3,
        max_workers: int = 5,
        skip_num_episodes: int = 0,
    ):
        self.podcast = podcast
        self.max_episodes = max_episodes
        self.max_workers = max_workers
        self.skip_num_episodes = skip_num_episodes

    def run_podcast(self):

        # TODO:
        # - √ clean up PodcastEpisode properties (too heavy)
        # - √ Add Transcript node
        # - create embeddings of all text
        #   > on paragraphs (text, one_sentence, few_sentences)
        #   > on person (name + description)
        #   > on podcast_episode (title + description)
        #   > on rels
        # - √ Add entities, edges
        # - add stats (word count, sentence count)
        # - √ enrich Person nodes with DBPedia URIs
        # - add audio clips (later)
        # - cleanup paragraphs from PodcatEpisode properties
        # - use pydantic models

        episodes = self.podcast.PodcastEpisodes
        if not episodes:
            episodes = self.podcast.load_episodes()

        logging.info(
            f"Pipeline episodes: {len(episodes)}. Handling {self.max_episodes} episodes."
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self.run_episode_tasks, episode)
                for episode in episodes[
                    self.skip_num_episodes : self.max_episodes
                ]
            ]

        concurrent.futures.wait(futures)

    def run_episode_tasks(self, episode: PodcastEpisode):

        try:

            episode.load()  # Load existing episode properties if available

            # download
            episode.download()

            # transcribe
            episode.transcribe()

            # clean up transcription / diarization
            episode.summarize_transcript()

            # define speakers
            episode.define_speakers()

            # create Person nodes for speakers
            episode.create_speakers()

            # create Paragraph nodes from transcription
            episode.paragraphize()

            # extract entities
            episode.extract_entities()

            # create stats
            # episode.create_stats()

            # create embeddings for all text
            episode.create_embeddings(rerun=True)

            # create statistics

        except Exception as e:
            logging.exception(f"Error processing episode {episode.url}: {e}")
            logging.warning("Episode was not saved to graph.")


def merge_duplicate_persons(batch_size=1, max_workers=1):
    """Merge duplicate nodes with the following keeper selection:
    1. Highest ID node WITH relationships is keeper
    2. If none have relationships, fall back to highest ID

    Will merge all relationships from duplicates to the keeper.
    Will delete all duplicate nodes.
    Will keep the properties of the keeper.
    """

    # 1. Find all duplicate groups
    dup_query = """
    MATCH (p)
    WHERE p.dbpedia_uri IS NOT NULL
      AND p.dbpedia_uri <> ''
    WITH p.dbpedia_uri AS uri, COLLECT(p) AS entities
    WHERE size(entities) > 1
    RETURN uri, entities
    """
    records, _, _ = memgraph.execute_query(dup_query)
    print(f"Found {len(records)} duplicate groups.")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(_merge_record, record)
            for record in records[:batch_size]
        ]

    concurrent.futures.wait(futures)


def _merge_record(record):
    """Merge duplicate Person nodes in batches."""

    uri = record["uri"]
    entities = record["entities"]
    print(f"\nProcessing duplicate group with DBpedia URI: {uri}")

    candidates = []
    # get edges of all entities
    for entity in entities:
        print(f"Entity ID: {entity.get("id")}, Name: {entity.get("name")}")
        query = """
        MATCH (p)-[r]-(other)
        WHERE id(p) = $entity_id
        RETURN
            id(r) AS rel_id,
            type(r) AS rel_type,
            properties(r) AS rel_props,
            properties(other) AS target_props
        """
        out_records, _, _ = memgraph.execute_query(
            query, {"entity_id": entity.id}
        )

        candidates.append(
            f"""
            Entity ID: {entity.id}
            Entity name: {entity.get("name")}
            Entity properties: {entity._properties}
            Relationships: {len(out_records)}
            """
        )

    print("\n".join(candidates))

    # extract entities using dspy
    with dspy.context(lm=strong_lm):
        logging.info(f"Determining keeper_id with strong model")
        extractor = dspy.ChainOfThought(DetermineDuplicateToKeep)
        pred = extractor(context="\n".join(candidates))
        print(pred)

    keeper_id = int(pred.keeper_id)
    if not keeper_id:
        print("No keeper ID found in prediction.")
        quit()
        # continue

    keeper = next((p for p in entities if p.id == keeper_id), None)
    if not keeper:
        print(f"Keeper with ID {keeper_id} not found in entities.")
        quit()

    print(f"Selected keeper: {keeper.id} - {keeper.get('name', 'Unknown')}")

    duplicates = [p for p in entities if p.id != keeper_id]
    print(f"Duplicates to merge: {[dup.id for dup in duplicates]}")

    # quit()

    for dup in duplicates:
        # 3. Transfer OUTGOING relationships (dup -> other)
        out_query = """
        MATCH (dup)-[r]->(other)
        WHERE id(dup) = $dup_id
        RETURN id(r) AS rel_id, type(r) AS rel_type, properties(r) AS props, id(other) AS other_id
        """
        out_records, _, _ = memgraph.execute_query(
            out_query, {"dup_id": dup.id}
        )
        print(f"Found {len(out_records)} outgoing relationships.")

        for rel in out_records:
            print(f"Processing relationship: {rel}")
            # Escape backticks in relationship type
            safe_rel_type = rel["rel_type"].replace("`", "``")
            create_out = f"""
            MATCH (keeper), (other)
            WHERE id(keeper) = $keeper_id AND id(other) = $other_id
            CREATE (keeper)-[r_new:`{safe_rel_type}`]->(other)
            SET r_new = $props
            """
            print(f"Creating relationship: {create_out}")
            print(f"With properties: {rel['props']}")
            print(f"Keeper ID: {keeper.id}, Other ID: {rel['other_id']}")
            print(f"Relationship type: {safe_rel_type}")
            print(f"Relationship properties: {rel['props']}")

            memgraph.execute_query(
                create_out,
                {
                    "keeper_id": keeper.id,
                    "other_id": rel["other_id"],
                    "props": rel["props"],
                },
            )

            # Delete old relationship by ID
            memgraph.execute_query(
                """
            MATCH ()-[r]->() 
            WHERE id(r) = $rel_id 
            DELETE r
            """,
                {"rel_id": rel["rel_id"]},
            )

        # 4. Transfer INCOMING relationships (other -> dup)
        in_query = """
        MATCH (other)-[r]->(dup)
        WHERE id(dup) = $dup_id
        RETURN id(r) AS rel_id, type(r) AS rel_type, properties(r) AS props, id(other) AS other_id
        """
        in_records, _, _ = memgraph.execute_query(in_query, {"dup_id": dup.id})

        for rel in in_records:
            # Escape backticks in relationship type
            safe_rel_type = rel["rel_type"].replace("`", "``")
            create_in = f"""
            MATCH (other), (keeper)
            WHERE id(other) = $other_id AND id(keeper) = $keeper_id
            CREATE (other)-[r_new:`{safe_rel_type}`]->(keeper)
            SET r_new = $props
            """
            memgraph.execute_query(
                create_in,
                {
                    "keeper_id": keeper.id,
                    "other_id": rel["other_id"],
                    "props": rel["props"],
                },
            )

            # Delete old relationship by ID
            memgraph.execute_query(
                """
            MATCH ()-[r]->() 
            WHERE id(r) = $rel_id 
            DELETE r
            """,
                {"rel_id": rel["rel_id"]},
            )

        # 5. Delete the duplicate node
        memgraph.execute_query(
            "MATCH (n) WHERE id(n) = $id DELETE n", {"id": dup.id}
        )

        # Mark the keeper as updated
        memgraph.execute_query(
            """
            MATCH (p)
            WHERE id(p) = $keeper_id
            SET p.updated_at = datetime(),
                p.ner_version = $ner_version
            """,
            {"keeper_id": keeper.id, "ner_version": NER_VERSION},
        )


def init_memgraph():
    """
    Initialize Memgraph database with constraints and indexes.
    """
    with memgraph.session() as session:
        # Create constraint for Podcast
        # TODO: contraint on dbpedia_uri for nodes
        session.run("CREATE CONSTRAINT ON (p:Podcast) ASSERT p.url IS UNIQUE")
        session.run(
            "CREATE CONSTRAINT ON (pe:PodcastEpisode) ASSERT pe.url IS UNIQUE"
        )

        # index
        session.run("CREATE INDEX ON :Person")
        session.run("CREATE INDEX ON :Person(name)")
        session.run("CREATE INDEX ON :Transcript")
        session.run("CREATE INDEX ON :Transcript(podcast_episode_url)")
        session.run("CREATE INDEX ON :Paragraph")
        session.run("CREATE INDEX ON :Paragraph(podcast_episode_url)")
        session.run("CREATE INDEX ON :Paragraph(start)")
        session.run("CREATE INDEX ON :Paragraph(end)")
        session.run("CREATE INDEX ON :PodcastEpisode")
        session.run("CREATE INDEX ON :PodcastEpisode(url)")

        # vector index
        session.run(
            f"""CREATE VECTOR INDEX vidx_paragraph_text ON :Paragraph(em_text) WITH CONFIG {{"dimension": 512, "capacity": 1024}};"""
        )
        session.run(
            f"""CREATE VECTOR INDEX vidx_paragraph_fews ON :Paragraph(em_few_sentences) WITH CONFIG {{"dimension": 512, "capacity": 1024}};"""
        )
        session.run(
            f"""CREATE VECTOR INDEX vidx_paragraph_ones ON :Paragraph(em_one_sentence) WITH CONFIG {{"dimension": 512, "capacity": 1024}};"""
        )


def clear_memgraph():
    """
    Clear all nodes and edges in Memgraph database.
    """
    with memgraph.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        logging.info("Cleared all nodes and edges in Memgraph.")


# def debug_llm():
#     """Debug LLM functionality."""
#     from dspy import ChainOfThought

#     # Example usage of dspy
#     reply = lm("")
#     print("LLM Reply:", reply)


def main():
    # init_memgraph()
    # no! clear_memgraph()

    # debug_clean_up_speakers_in_paragraph_nodes()
    # add_dbpedia_uris_to_persons()
    # debug_llm()
    # return

    # return merge_duplicate_persons(batch_size=10, max_workers=1)

    podcast = Podcast(url="https://www.youtube.com/@GDiesen1/videos")
    podcast.fetch_meta()
    podcast.save()
    podcast.load()
    episodes = podcast.load_episodes()
    logging.info(f"Podcast episodes: {len(episodes)}")

    # for episode in episodes:
    # episode.debug_clean_up_speakers()

    pipeline = Pipeline(
        podcast, max_episodes=4, max_workers=1, skip_num_episodes=3
    )
    pipeline.run_podcast()

    # episode = podcast.get_episode(
    #     url="https://www.youtube.com/watch?v=SWux-RBbKGs"
    # )
    # if episode:
    #     pipeline.run_episode_tasks(episode)


if __name__ == "__main__":
    main()
