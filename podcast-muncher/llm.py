import dspy

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
