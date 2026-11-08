"""Try Piiranha anonymisation. Usage: uv run python try_piiranha.py ["sentence" ...]  (no args = interactive)"""
import sys

from transformers import pipeline

MODEL = "iiiorg/piiranha-v1-detect-personal-information"

# Piiranha label -> our label (see SCOPING.md section 4); anything else is dropped
LABELS = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON",
    "CITY": "LOCATION", "STREET": "LOCATION", "BUILDINGNUM": "LOCATION", "ZIPCODE": "LOCATION",
    "TELEPHONENUM": "PHONE_NUMBER",
    "SOCIALNUM": "SSN", "IDCARDNUM": "SSN", "TAXNUM": "SSN",
    "EMAIL": "EMAIL_ADDRESS",
    "ACCOUNTNUM": "IBAN",
    "DATEOFBIRTH": "DATE_TIME",
}

ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")


def anonymize(text):
    spans = []
    for s in ner(text):
        # tokens carry the preceding space and sometimes a trailing comma/period: trim both
        chunk = text[s["start"] : s["end"]]
        s["start"] += len(chunk) - len(chunk.lstrip())
        s["end"] = s["start"] + len(chunk.strip().rstrip(".,;:!?"))
        print(f"  {s['entity_group']:14} {s['score']:.2f}  {text[s['start']:s['end']]!r}")
        if s["entity_group"] in LABELS:
            label = LABELS[s["entity_group"]]
            prev = spans[-1] if spans else None
            # merge adjacent same-label spans separated only by whitespace (given name + surname)
            if prev and prev[2] == label and not text[prev[1] : s["start"]].strip():
                prev[1] = s["end"]
            else:
                spans.append([s["start"], s["end"], label])
    for start, end, label in reversed(spans):
        text = text[:start] + f"<{label}>" + text[end:]
    return text


sentences = sys.argv[1:]
if sentences:
    for t in sentences:
        print(anonymize(t))
else:
    while t := input("\nsentence (empty = quit)> ").strip():
        print(anonymize(t))
