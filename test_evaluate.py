"""Small checks: gold alignment, validation, the token conversion + scoring, anonymize.

Run: uv run python test_evaluate.py   (or pytest)
"""
from pathlib import Path

from anonymize import anonymize
from detectors import Entity
from evaluate import DATA, gold_spans, load_benchmark, score


def test_gold_alignment():
    g = gold_spans("Call Bob in Paris.", "Call <PERSON> in <LOCATION>.")
    assert [(e.text, e.label, e.start, e.end) for e in g] == [("Bob", "PERSON", 5, 8), ("Paris", "LOCATION", 12, 17)]


def test_bad_rows_fail():
    for text, labelled in [("Call Bob", "Call <PERSN>"), ("Call Bob", "Ring <PERSON>")]:
        try:
            gold_spans(text, labelled)
        except ValueError:
            continue
        raise AssertionError(f"accepted {labelled!r}")


def test_all_benchmarks_load():
    for path in DATA.glob("*.csv"):
        assert load_benchmark(path), path


def test_scoring_perfect_and_empty():
    examples = load_benchmark(DATA / "benchmark.csv")
    by_text = {ex["text"]: ex["gold"] for ex in examples}

    class Perfect:
        def detect(self, text):
            return by_text[text]

    class Nothing:
        def detect(self, text):
            return []

    perfect, nothing = score(examples, Perfect()), score(examples, Nothing())
    assert perfect["micro"]["f2"] == 1.0 and perfect["errors"] == []
    assert nothing["micro"]["recall"] == 0.0 and nothing["micro"]["support"] == perfect["micro"]["support"]


def test_anonymize():
    assert anonymize("Hi Bob", [Entity("Bob", "PERSON", 3, 6)]) == "Hi <PERSON>"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
