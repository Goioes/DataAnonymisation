# Data Anonymization

Extract sensitive entities from text with pre-trained NER models (GLiNER, Piiranha), anonymize them with `<LABEL>` placeholders, and benchmark both models on `Data/benchmark.csv`. Scope and decisions: [SCOPING.md](SCOPING.md).

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```
uv sync
```

Model weights are downloaded from Hugging Face on first run (not committed).

## Usage
_TODO: demo app, benchmark script, API._

## Method
_TODO: models, label mapping, matching (presidio-evaluator, IoU 0.5, F2)._

## Results
_TODO: per-label table._

## Dataset caveats
_TODO: see SCOPING.md section 7._

## Licences
_TODO: Piiranha (CC-BY-NC-ND 4.0, non-commercial), GLiNER multi (Apache-2.0); verify on model cards._
