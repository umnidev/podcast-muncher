import dspy


class CleanTranscript(dspy.Signature):
    """
    Clean up transcript.
    - Keep it as close to original as possible.
    - Fix grammatical errors and transcription errors.
    - Make sure NOT to leave out ANY significant detail from the transcript, ie. names, meanings, numbers.

    Return the cleaned, unabridged transcript, and two condensed versions:
    - one_sentence: Text condensed to a single sentence, as if the speaker said only this sentence.
    - few_sentences: Text condensed to maximum three sentences, as if the speaker said only this.
    """

    speech: str = dspy.InputField()
    context: str = dspy.InputField()

    text: str = dspy.OutputField(
        desc="Cleaned, unabridged transcript of the speech."
    )
    one_sentence: str = dspy.OutputField(
        desc="Text condensed to a single sentence, as if the speaker said only this sentence. "
    )
    few_sentences: str = dspy.OutputField(
        desc="Text condensed to maximum three sentences, as if the speaker said only this. "
    )


class DefineSpeakers(dspy.Signature):
    """
    Extract speakers from the description and partial transcript of a podcast episode.
    The transcript only contains speaker_id, eg. SPEAKER_01, SPEAKER_02, etc.
    We want to extract the names, roles, and descriptions of the speakers based on
    the context provided in the transcript, and importantly, match the speaker_id to the names.

    Return a list of speaker dicts, each containing:
    - last_name: Last name of the speaker
    - first_name: First name of the speaker
    - name: Full name of the speaker (last_name + first_name)
    - title: Title of the speaker (optional)
    - title_short: Short title of the speaker (optional)
    - role: Role of the speaker ('host' or 'guest')
    - speaker_id: Unique identifier for the speaker in the transcript, eg. SPEAKER_01
    - description: Short description of the speaker (optional)
    - description_long: Long description of the speaker (optional)
    - subsctack_url: URL to the speaker's Substack profile (optional)
    - linkedin_url: URL to the speaker's LinkedIn profile (optional)
    - twitter_url: URL to the speaker's Twitter profile (optional)
    - youtube_url: URL to the speaker's YouTube profile/channel (optional)
    - patreon_url: URL to the speaker's Patreon profile (optional)
    - paypal_url: URL to the speaker's PayPal profile (optional)
    """

    context: str = dspy.InputField(
        desc="Description and partial transcript of the podcast episode."
    )
    speakers: list[dict] = dspy.OutputField(
        desc="List of speaker dicts extracted from the transcript."
    )


class ExtractEntities(dspy.Signature):
    """
    Do Named Entity Extraction (NER) on text.
    Use DBPedia ontology for entities.

    For Person type entities, add a "details" key with best guesses on full name.

    For relations, don't include entites in relation label, and use natural words, eg.
    with text "China invested heavily in Sudan", relation label becomes "invested heavily in".
    It's extremely important that ENTITIES are not included in the relation label.

    Only create relations with high confidence.

    ---
    Example 1:

    INPUT:
    text: "Most importantly, Russia recently captured Bakhmut. Trump harshly critized Russia."

    OUTPUT:
    entities: [
        {
            "ner_id": "1",              // id for this extraction session only, serial
            "text": "Russia",           // source word in text
            "type": "country",          // simple entity type
            "dbo_type": "dbo:Country",  // DBPedia Ontology type
            "name": "Russia",           // entity name
            "confidence": 0.99,         // confidence in entity recognition
            "dbpedia_uri": "https://dbpedia.org/page/Russia" // dbpedia uri, if known
        },
        {
            "ner_id": "2",          // id for this extraction session only, serial
            "text": "Bakhmut",      // source word in text
            "type": "city",         // entity type
            "dbo_type": "dbo:City", // DBPedia Ontology type
            "name": "Bakhmut",      // entity name
            "confidence": 0.94,     // confidence in entity recognition
            "dbpedia_uri": "https://dbpedia.org/page/Bakhmut" // dbpedia uri, if known
        },
        {
            "ner_id": "3",              // id for this extraction session only, serial
            "text": "Trump",            // exact source word in text
            "type": "person",           // entity type
            "dbo_type": "dbo:Person",   // DBPedia Ontology type
            "name": "Donald Trump",     // entity name (based on direct evidence in text) (full name if Person type)
            "last_name": "Trump",       // best guesstimate of last_name (Person type only)
            "first_name": "Donald",     // best guesstimate of first_name (Person type only)
            "dbpedia_uri": "https://dbpedia.org/page/Donald_Trump" // dbpedia uri, if known
            "confidence": 0.94,         // confidence in entity recognition
        },
    ]

    relations: [
        {
            "source": "1", // ner_id from above (Russia)
            "target": "2", // ner_id from entities above (Bakhmut)
            "name": "recently captured", // relation name
            "description": "Russia recently captured Bakhmut", // description of the relation
            "dbo_type": "dbo:Capture", // DBPedia Ontology type for the relation (if applicable)
            "confidence": 0.92, // confidence in relation recognition
        },
        {
            "source": "3", // ner_id from above (Donald Trump)
            "target": "1", // ner_id from entities above (Russia)
            "name": "harshly critized", // relation name
            "description": "Trump harshly critized Russia", // description of the relation
            "dbo_type": "dbo:Criticism", // DBPedia Ontology type for the relation (if applicable)
            "confidence": 0.92, // confidence in relation recognition
        }
    ]

    """

    text: str = dspy.InputField(desc="Text from which to extract entities")
    entities: list[dict] = dspy.OutputField(desc="Extracted entities")
    relations: list[dict] = dspy.OutputField(desc="Extracted relations")


class AssignDBPediaUri(dspy.Signature):
    """
    Assign the correct DBPedia URI for each entity.
    """

    context: str = dspy.InputField(
        desc="Entity for which to assign DBPedia URI. Includes name, description."
    )
    dbpedia_uri: str = dspy.OutputField(
        desc="DBPedia URI, eg. 'https://dbpedia.org/page/Julian_Assange'"
    )


class DetermineDuplicateToKeep(dspy.Signature):
    """
    Determine which duplicate entity to keep based on the context.
    The context is a list of entities with their properties, such as name, description, etc.
    The function should return the entity to keep by ID.

    Criteria in order of importance:
    1. Highest confidence in dbo_type
    2. Highest ID (most recent entity), unless no relations to other entities.
    3. Highest confidence in entity recognition.
    4. Highest number of relations.
    5. Most complete description.

    Return:
    - keeper_id: Entity ID to keep, based on the criteria above.
    """

    context: list[dict] = dspy.InputField(
        desc="List of entities with their properties."
    )
    keeper_id: int = dspy.OutputField(desc="Entity to keep")
