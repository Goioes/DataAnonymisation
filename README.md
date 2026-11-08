# Data Anonymization

Extract sensitive entities from text with pre-trained NER models (GLiNER, Piiranha), anonymize them with `<LABEL>` placeholders, and benchmark both models on `Data/benchmark.csv`. Scope and decisions: [SCOPING.md](SCOPING.md).

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```
uv sync
```

Model weights are downloaded from Hugging Face on first run (not committed).

## Usage
Demo app (pick an example or paste text, choose a model, click Anonymize):

```
uv run streamlit run app.py
```

Command-line checks (default sentence, or pass your own; set `PYTHONUTF8=1` on Windows):

```
uv run python detectors.py "sentence"   # entities per model
uv run python anonymize.py "sentence"   # anonymized text per model
```

_TODO: benchmark script, API._

**Known limitation:** there is no length handling yet. Text beyond a model's token limit may be left unprocessed, so the "anonymized" output can still contain sensitive data.

## Method
_TODO: models, label mapping, matching (presidio-evaluator, IoU 0.5, F2)._

## Results
_TODO: per-label table._

## Dataset caveats
_TODO: see SCOPING.md section 7._

## Licences
_TODO: Piiranha (CC-BY-NC-ND 4.0, non-commercial), GLiNER `gliner_large-v2.5` (Apache-2.0); verify Piiranha on its model card._
