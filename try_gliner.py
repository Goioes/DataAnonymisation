"""Try GLiNER anonymisation. Usage: uv run python try_gliner.py ["sentence" ...]  (no args = interactive)"""
import sys

from gliner import GLiNER

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
    "iban": "IBAN",
    "social security number": "SSN",
}

model = GLiNER.from_pretrained(MODEL, load_tokenizer=True)


def anonymize(text):
    spans = model.predict_entities(text, list(LABELS), threshold=THRESHOLD)
    for s in spans:
        print(f"  {s['label']:24} {s['score']:.2f}  {s['text']!r}")
    # replace right-to-left so earlier offsets stay valid
    for s in sorted(spans, key=lambda s: s["start"], reverse=True):
        text = text[: s["start"]] + f"<{LABELS[s['label']]}>" + text[s["end"] :]
    return text


sentences = sys.argv[1:]
if sentences:
    for t in sentences:
        print(anonymize(t))
else:
    while t := input("\nsentence (empty = quit)> ").strip():
        print(anonymize(t))
