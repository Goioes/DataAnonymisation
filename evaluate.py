"""Benchmark: score the selected detectors on the selected benchmarks (Data/*.csv, `text;label`).

Run:   uv run python evaluate.py     (menus for benchmarks and models)
Check: uv run python test_evaluate.py
Writes results/<benchmark>/<model>.json, overwriting that pair only.
"""
import json
import re
import string
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from detectors import ALL_LABELS, DETECTORS, Entity, get_detector

DATA = Path("Data")
RESULTS = Path("results")
IOU_THRESHOLD = 0.5
# Language-neutral on purpose (SCOPING.md section 7): whitespace and punctuation only. Not the Presidio
# default, which strips English words (and "de") and so is lenient in a different way per language.
SKIP_WORDS = [""] + list(string.punctuation + "«»“”‘’„‚–—…¿¡")
PLACEHOLDER = re.compile(r"<([A-Z_]+)>")


def gold_spans(text, labelled):
    """Recover gold spans: literal text is kept, each <LABEL> in `labelled` becomes a capture group."""
    parts = PLACEHOLDER.split(labelled)  # literal, label, literal, label, ..., literal
    labels = parts[1::2]
    unknown = [lab for lab in labels if lab not in ALL_LABELS]
    if unknown:
        raise ValueError(f"unknown label(s) {unknown}")
    pattern = "".join(re.escape(p) if i % 2 == 0 else "(.+?)" for i, p in enumerate(parts))
    m = re.fullmatch(pattern, text, re.S)
    if not m:
        raise ValueError("labelled text does not align with text")
    return [Entity(m.group(i + 1), lab, m.start(i + 1), m.end(i + 1)) for i, lab in enumerate(labels)]


def load_benchmark(path):
    """Validate every row first; one error listing all bad rows. Extra columns (e.g. language) pass through."""
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError(f"{path}: needs columns 'text' and 'label', found {list(df.columns)}")
    examples, errors = [], []
    for i, row in df.iterrows():
        try:
            gold = gold_spans(row["text"], row["label"])
        except ValueError as e:
            errors.append(f"{path} row {i + 2}: {e}")
            continue
        examples.append({"text": row["text"], "gold": gold, "meta": row.drop(["text", "label"]).to_dict()})
    if errors:
        raise ValueError("\n".join(errors))
    return examples


def token_frame(sample_id, text, gold, predicted, nlp):
    """Per-token tags, the format SpanEvaluator expects. A token takes the first entity it overlaps."""
    tokens = list(nlp(text))

    def tags(entities):
        out = ["O"] * len(tokens)
        for e in sorted(entities, key=lambda e: e.start):
            for j, t in enumerate(tokens):
                if out[j] == "O" and t.idx < e.end and t.idx + len(t) > e.start:
                    out[j] = e.label
        return out

    return pd.DataFrame({
        "sentence_id": sample_id,
        "token": [t.text for t in tokens],
        "start_indices": [t.idx for t in tokens],
        "annotation": tags(gold),
        "prediction": tags(predicted),
    })


def prf(tp, predicted, support):
    p = tp / predicted if predicted else 0.0
    r = tp / support if support else 0.0
    f = lambda b: (1 + b * b) * p * r / (b * b * p + r) if p + r else 0.0
    return {"support": support, "predicted": predicted, "tp": tp,
            "precision": p, "recall": r, "f1": f(1), "f2": f(2)}


def score(examples, detector):
    import spacy  # heavy imports only when scoring
    from presidio_evaluator.evaluation import SpanEvaluator

    nlp = spacy.blank("xx")  # tokenizer only, no model download, no language-specific rules
    ev = SpanEvaluator(skip_words=SKIP_WORDS, iou_threshold=IOU_THRESHOLD, char_based=True)
    frames, errors, details, res = [], [], [], None
    for i, ex in enumerate(examples):
        predicted = [e for e in detector.detect(ex["text"]) if e.label in ALL_LABELS]  # safety net
        details.append({"text": ex["text"], "gold": [asdict(e) for e in ex["gold"]], "predicted": [asdict(e) for e in predicted]})
        frame = token_frame(i, ex["text"], ex["gold"], predicted, nlp)
        frames.append(frame)
        seen = len(res.model_errors or []) if res else 0
        res = ev.calculate_score_on_df(frame, level="entity", evaluation_result=res)
        errors += [{"example": i, "type": str(getattr(e.error_type, "value", e.error_type)),
                    "gold": e.annotation, "predicted": e.prediction, "text": e.full_text, **ex["meta"]}
                   for e in res.model_errors[seen:]]
    pii = ev.calculate_score_on_df(pd.concat(frames, ignore_index=True), level="pii")

    zero = type("Zero", (), {"true_positives": 0, "num_predicted": 0, "num_annotated": 0})()
    per = {lab: res.per_type.get(lab, zero) for lab in ALL_LABELS} if res else {}
    return {
        "examples": len(examples),
        "labels": {lab: prf(m.true_positives, m.num_predicted, m.num_annotated) for lab, m in per.items()},
        "micro": prf(sum(m.true_positives for m in per.values()), sum(m.num_predicted for m in per.values()),
                     sum(m.num_annotated for m in per.values())),
        "global": prf(pii.pii_true_positives, pii.pii_predicted, pii.pii_annotated),  # type-agnostic PII vs O
        "errors": errors,
        "details": details,  # per example: text, gold and predicted spans (for the Benchmark page)
    }


def choose(title, options):
    print(f"{title}:\n  0) all")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        try:
            picks = {int(x) for x in input("Pick (e.g. 1,2): ").split(",")}
            if 0 in picks:
                return list(options)
            if picks and all(1 <= p <= len(options) for p in picks):
                return [options[p - 1] for p in sorted(picks)]
        except ValueError:
            pass
        print("Not a valid choice.")


def main():
    names = sorted(p.stem for p in DATA.glob("*.csv"))
    benchmarks = choose("Benchmarks", names)
    models = choose("Models", list(DETECTORS))
    try:
        loaded = {b: load_benchmark(DATA / f"{b}.csv") for b in benchmarks}  # validate before any model loads
    except ValueError as e:
        sys.exit(f"Invalid benchmark, nothing scored:\n{e}")
    for model in models:
        detector = get_detector(model)
        for b, examples in loaded.items():
            result = {"benchmark": b, "model": model, "scored_at": datetime.now().isoformat(timespec="seconds"),
                      **score(examples, detector)}
            out = RESULTS / b / f"{model}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            m = result["micro"]
            print(f"{b} / {model}: micro P={m['precision']:.2f} R={m['recall']:.2f} F2={m['f2']:.2f} -> {out}")


if __name__ == "__main__":
    main()
