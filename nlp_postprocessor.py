# nlp_postprocessor.py

import warnings

try:
    import language_tool_python
    _HAS_LANGUAGE_TOOL = True
except ImportError:
    language_tool_python = None
    _HAS_LANGUAGE_TOOL = False

class NLPPostProcessor:
    # Example accent-specific corrections
    ACCENT_CORRECTIONS = {
        # Arabic accent common errors
        "ze": "the",
        "zis": "this",
        "beoble": "people",
        # South Asian accent example
        "wery": "very",
        # Add more based on Ammar's error analysis
    }

    def __init__(self):
        if _HAS_LANGUAGE_TOOL:
            self.tool = language_tool_python.LanguageTool('en-US')
        else:
            self.tool = None

    def correct_grammar(self, text):
        """Use language-tool-python to correct grammar mistakes."""
        if not self.tool:
            return text
        matches = self.tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected

    def restore_punctuation(self, text):
        """Restore capitalization and punctuation in the text."""
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def apply_accent_corrections(self, text):
        """Fix common accent-based misrecognitions."""
        words = text.split()
        corrected = []
        for word in words:
            lower = word.lower().strip('.,!?')
            if lower in self.ACCENT_CORRECTIONS:
                corrected.append(self.ACCENT_CORRECTIONS[lower])
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def process(self, result):
        """
        Apply all NLP post-processing to transcription result.
        `result` is a dictionary with "segments" and "text".
        """
        for segment in result["segments"]:
            text = segment["text"]
            text = self.apply_accent_corrections(text)
            text = self.correct_grammar(text)
            text = self.restore_punctuation(text)
            segment["text"] = text

        # Update the full text
        result["text"] = ' '.join([s["text"] for s in result["segments"]])
        return result