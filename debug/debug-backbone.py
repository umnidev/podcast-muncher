from backbone import Backbone, Node, Edge, Property, PropertyType, Config, Ontology # pyright: ignore[reportMissingImports]
from pathlib import Path
import yaml
from pprint import pprint

print(Backbone)

config = Config(kuzu_data_path=str(Path(__file__).parent / ".test_data"))

backbone = Backbone(config=config)

backbone.clear_graph()

with open("../podcast-ontology.yaml", "r") as yaml_file:
    ontology_dict = yaml.safe_load(yaml_file)

pprint(ontology_dict)

ontology = Ontology(ontology_dict)

# ontology = Ontology(
#     {
#         'edges': [{'from': 'PodcastEpisode',
#             'name': 'CONTENT',
#             'properties': {'created_at': {'type': 'DATE'}},
#             'to': 'Paragraph'},
#            {'from': 'Paragraph',
#             'name': 'NEXT',
#             'properties': {'created_at': {'type': 'DATE'}},
#             'to': 'Paragraph'},
#            {'from': 'Paragraph',
#             'name': 'SPOKEN_BY',
#             'properties': {'created_at': {'type': 'DATE'}},
#             'to': 'Person'},
#            {'from': 'Person',
#             'name': 'IS_GUEST_ON',
#             'properties': {'created_at': {'type': 'DATE'}},
#             'to': 'PodcastEpisode'},
#            {'from': 'Person',
#             'name': 'IS_HOST_OF',
#             'properties': {'created_at': {'type': 'DATE'}},
#             'to': 'PodcastEpisode'}],
#         'nodes': [{'label': 'Podcast',
#             'properties': {'description': {'type': 'STRING'},
#                            'name': {'type': 'STRING'},
#                            'number_of_subscribers': {'type': 'INT64'},
#                            'youtube_channel_url': {'type': 'STRING'}}},
#            {'label': 'PodcastEpisode',
#             'properties': {'date': {'type': 'DATE'},
#                            'description': {'type': 'STRING'},
#                            'length_in_seconds': {'type': 'INT64'},
#                            'num': {'type': 'INT64'},
#                            'summary': {'type': 'STRING'},
#                            't11n_clean': {'type': 'STRING'},
#                            't11n_raw': {'type': 'STRING'},
#                            'title': {'type': 'STRING'},
#                            'youtube_video_url': {'type': 'STRING'}}},
#            {'label': 'Paragraph',
#             'properties': {'summary': {'type': 'STRING'},
#                            'text': {'type': 'STRING'},
#                            'timestamp_end': {'type': 'INT64'},
#                            'timestamp_start': {'type': 'INT64'}}},
#            {'label': 'Person',
#             'properties': {'dbpedia_uri': {'type': 'STRING'},
#                            'description': {'type': 'STRING'},
#                            'name': {'type': 'STRING'},
#                            'profession': {'type': 'STRING'}}}]
                           
# }
# )

print(f"Ontology: {ontology}")
backbone.set_ontology(ontology)
quit()
# 

# ontology = Ontology(
#     {
#         "nodes": [
#             {
#                 "label": "Author",
#                 "properties": {
#                     "id": {"type": "SERIAL", "primary_key": True},
#                     "name": {"type": "STRING"},
#                     "rating": {"type": "DOUBLE"},
#                 },
#             },
#             {
#                 "label": "Book",
#                 "properties": {
#                     "isbn": {"type": "STRING", "primary_key": True},
#                     "title": {"type": "STRING"},
#                     "year": {"type": "INT64"},
#                 },
#             },
#         ],
#         "edges": [
#             {
#                 "label": "wrote",
#                 "from": "Author",
#                 "to": "Book",
#                 "properties": {"royalty_percent": {"type": "DOUBLE"}},
#             }
#         ],
#     }
# )

backbone.set_ontology(ontology)

melville = Node(
    label="Author",
    properties={
        "name": Property(
            name="name", value="Herman Melville", type=PropertyType.STRING
        ),
        "rating": Property(name="rating", value=4.7, type=PropertyType.DOUBLE),
    },
)
added_author = backbone.graph.add_node(melville)


print(added_author)