"""Replace entities in a text with <LABEL> placeholders.

Check: uv run python anonymize.py ["sentence" ...]
"""
import sys

from detectors import DETECTORS, Entity, get_detector


def anonymize(text: str, entities: list[Entity]) -> str:
    # replace right-to-left so earlier offsets stay valid; skip an entity that overlaps one already replaced
    last_start = len(text)
    for e in sorted(entities, key=lambda e: e.start, reverse=True):
        if e.end > last_start:
            continue
        text = text[: e.start] + f"<{e.label}>" + text[e.end :]
        last_start = e.start
    return text


if __name__ == "__main__":
    sentences = sys.argv[1:] or ["Tim Cook is the CEO of Apple and lives in Cupertino. Mail him at tim@apple.com."]
    for name in DETECTORS:
        print(f"== {name}")
        for t in sentences:
            print(anonymize(t, get_detector(name).detect(t)))
