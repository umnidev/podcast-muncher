import dspy


class SmoothTranscription(dspy.Signature):
    """
    Minimally clean transcription. Make it more legible, but keep it as close to original as possible.
    Fix grammatical errors and transcription errors.
    Make sure NOT to leave out ANY significant detail from the transcript, ie. names, meanings.

    Also add short- and long versions of the transcript, as if the speaker were more succinct.
    """

    speech: str = dspy.InputField()
    context: str = dspy.InputField()
    text: str = dspy.OutputField()
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
    - full_name: Full name of the speaker (last_name + first_name)
    - title: Title of the speaker (optional)
    - title_short: Short title of the speaker (optional)
    - role: Role of the speaker (e.g., host, guest)
    - speaker_id: Unique identifier for the speaker in the transcript, eg. SPEAKER_01
    - description: Short description of the speaker (optional)
    """

    context: str = dspy.InputField(
        desc="Description and partial transcript of the podcast episode."
    )
    speakers: list[dict] = dspy.OutputField(
        desc="List of speaker dicts extracted from the transcript."
    )
