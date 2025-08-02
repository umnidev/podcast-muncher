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


def main():
    with open("../podcast-ontology.yaml", "r") as yaml_file:
        ontology_dict = yaml.safe_load(yaml_file)   

    config = Config(kuzu_data_path=str(Path(__file__).parent / "../data/podcast.kuzu"))
    ontology = Ontology(ontology_dict)
    backbone = Backbone(config)
    # backbone.clear_graph()
    backbone.set_ontology(ontology)
    logging.debug(backbone.get_ontology().as_yaml())


    podcast = backbone.graph.add_node(Node(
        label="Podcast",
        properties={
            "url": Property(
                name="url",
                value="debug:123456",
                type=PropertyType.STRING
            )
        }
    ))
    episode = backbone.graph.add_node(Node(
        label="PodcastEpisode",
        properties={
            "url": Property(
                name="url",
                value="debug:123456",
                type=PropertyType.STRING
            )
        }
    ))

    edge = backbone.graph.add_edge(Edge(
        label="HAS_EPISODE",
        from_node=podcast,
        to_node=episode
    ))

    pprint(edge)

if __name__ == "__main__":
    main()
