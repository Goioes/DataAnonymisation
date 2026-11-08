"""Detectors: one wrapper per model, each returns Entity spans in our 12-label schema.

Check: uv run python detectors.py ["sentence" ...]
"""
import sys
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


class GlinerDetector:
    MODEL = "gliner-community/gliner_large-v2.5"
    THRESHOLD = 0.5
    # natural-language label wording passed to GLiNER -> our label
    LABELS = {
        "person": "PERSON",
        "organization": "ORG",
        "job title": "JOB",
        "email address": "EMAIL_ADDRESS",
        "location": "LOCATION",
        "monetary amount": "AMOUNT",
        "date": "DATE_TIME",
        "university": "UNIVERSITY",
        "phone number": "PHONE_NUMBER",
        "url": "URL",
        "bank account": "IBAN",
        "social security number": "SSN",
    }

    def __init__(self):
        from gliner import GLiNER  # heavy import, only when the model is first used

        self.model = GLiNER.from_pretrained(self.MODEL, load_tokenizer=True)

    def detect(self, text):
        spans = self.model.predict_entities(text, list(self.LABELS), threshold=self.THRESHOLD)
        return [Entity(s["text"], self.LABELS[s["label"]], s["start"], s["end"]) for s in spans]


class PiiranhaDetector:
    MODEL = "iiiorg/piiranha-v1-detect-personal-information"
    # source label -> our label (SCOPING.md section 4); anything else is dropped
    LABELS = {
        "GIVENNAME": "PERSON", "SURNAME": "PERSON",
        "CITY": "LOCATION", "STREET": "LOCATION", "BUILDINGNUM": "LOCATION", "ZIPCODE": "LOCATION",
        "TELEPHONENUM": "PHONE_NUMBER",
        "SOCIALNUM": "SSN", "IDCARDNUM": "SSN", "TAXNUM": "SSN",  # last two: still to test
        "EMAIL": "EMAIL_ADDRESS",
        "ACCOUNTNUM": "IBAN",
        "DATEOFBIRTH": "DATE_TIME",
        "USERNAME": "USERNAME",
        "PASSWORD": "PASSWORD"
    }

    def __init__(self):
        from transformers import pipeline

        self.ner = pipeline("token-classification", model=self.MODEL, aggregation_strategy="simple")

    def detect(self, text):
        spans = []  # [start, end, label]
        for s in self.ner(text):
            if s["entity_group"] not in self.LABELS:
                continue
            # tokens carry the preceding space and sometimes a trailing comma/period: trim both
            chunk = text[s["start"] : s["end"]]
            start = s["start"] + len(chunk) - len(chunk.lstrip())
            end = start + len(chunk.strip().rstrip(".,;:!?"))
            if end <= start:  # token was only punctuation
                continue
            label = self.LABELS[s["entity_group"]]
            # merge adjacent same-label spans separated only by whitespace (given name + surname)
            if spans and spans[-1][2] == label and not text[spans[-1][1] : start].strip():
                spans[-1][1] = end
            else:
                spans.append([start, end, label])
        return [Entity(text[start:end], label, start, end) for start, end, label in spans]


DETECTORS = {"GLiNER": GlinerDetector, "Piiranha": PiiranhaDetector}


@cache
def get_detector(name):
    """Load a detector on first use; later calls return the same instance."""
    return DETECTORS[name]()


if __name__ == "__main__":
    sentences = sys.argv[1:] or ["Tim Cook is the CEO of Apple and lives in Cupertino. Mail him at tim@apple.com."]
    for name in DETECTORS:
        print(f"== {name}")
        for t in sentences:
            for e in get_detector(name).detect(t):
                print(f"  {e.label:14} {e.start:4}-{e.end:<4} {e.text!r}")
