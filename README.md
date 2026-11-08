# Data Anonymization

Extract sensitive entities from text with pre-trained NER models (GLiNER, Piiranha), anonymize them with `<LABEL>` placeholders, and benchmark both models on the labelled datasets in `Data/`.

## Installation
1. **Install [uv](https://docs.astral.sh/uv/)** (Python package and project manager). Skip if `uv --version` already works.

   Windows (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   macOS / Linux:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Open a new terminal afterwards so `uv` is on your PATH.

2. **Clone the repository:**
   ```
   git clone https://github.com/Goioes/DataAnonymisation.git
   cd DataAnonymisation
   ```

3. **Install the dependencies:**
   ```
   uv sync
   ```
   This creates a `.venv` folder with all packages from `uv.lock`. You don't need Python installed beforehand: if Python 3.12 is missing, uv downloads it (version pinned in `.python-version`).

4. **Download the models and check the install:**
   ```
   uv run python detectors.py
   ```
   The first run downloads the GLiNER and Piiranha weights from Hugging Face (a few GB, cached in `~/.cache/huggingface`, not committed) and prints the entities each model finds in a sample sentence. Later runs use the cache. Both the demo app and the API load all models at startup, so without this step their first start is slow.

## Usage

### Demo app
Pick an example or paste text, choose a model, click Anonymize. A second page shows the benchmark results.

```
uv run streamlit run app.py
```

### API
Start the server:

```
uv run uvicorn api:app
```

Endpoints: `GET /models` and `POST /anonymize` (`model` is optional and defaults to `GLiNER`; `text` is 1 to 1000 characters, otherwise 422). Interactive docs, where you can try requests in the browser: http://127.0.0.1:8000/docs

PowerShell:

```powershell
irm http://127.0.0.1:8000/anonymize -Method Post -ContentType application/json -Body '{"text": "Tim Cook is the CEO of Apple and lives in Cupertino.", "model": "GLiNER"}'
```

curl:

```
curl -X POST http://127.0.0.1:8000/anonymize -H "Content-Type: application/json" -d '{"text": "Tim Cook is the CEO of Apple and lives in Cupertino.", "model": "GLiNER"}'
```

Response: `{"anonymized_text": "...", "entities": [{"text", "label", "start", "end"}]}`.

### Benchmark
```
uv run python evaluate.py
```

Shows a menu to choose benchmarks (every `Data/*.csv`) and models, each with an "all" option. Results are written to `results/<benchmark>/<model>.json` and shown on the Benchmark page of the demo app.

### Command-line checks
Default sentence, or pass your own:

```
uv run python detectors.py "sentence"   # entities per model
uv run python anonymize.py "sentence"   # anonymized text per model
```

### Tests
```
uv run python test_evaluate.py
uv run python test_api.py
```

**Known limitation:** there is no length handling yet. Text beyond a model's token limit may be left unprocessed, so the "anonymized" output can still contain sensitive data. The API therefore rejects texts over 1000 characters.

## Method
Models, label mapping, how matching and scoring work, dataset caveats and how to add a model or benchmark: [METHOD.md](METHOD.md).

## Results
Entity-level, 14 labels, micro-averaged, IoU 0.5. The headline is F2 (recall weighted double: a missed entity is a leak). Two views: *detection + label* (strict: the PII must be found and get the right label) and *detection only* ("PII vs O": any PII found counts, the label is ignored, which is the closest thing to a leak rate). The gap between the two is the share of errors that are wrong labels. Details and caveats in [METHOD.md](METHOD.md); every example and error is on the Benchmark page of the demo app.

| Benchmark | Model | Precision (detection + label) | Recall (detection + label) | F2 (detection + label) | F2 (detection only) |
|---|---|---|---|---|---|
| `benchmark` (10 English examples) | GLiNER | 0.89 | 0.89 | **0.89** | 0.96 |
| `benchmark` | Piiranha | 0.80 | 0.17 | **0.21** | 0.21 |
| `benchmark_extra` (10 FR/NL/EN examples) | GLiNER | 0.89 | 0.89 | **0.89** | 0.94 |
| `benchmark_extra` | Piiranha | 0.85 | 0.32 | **0.37** | 0.39 |

Per label on `benchmark` (P / R / F2). Piiranha has no ORG, JOB, AMOUNT, UNIVERSITY or URL label, so those score 0 by construction. IBAN, SSN, USERNAME and PASSWORD have no gold examples in this benchmark and are left out (GLiNER produced one false USERNAME).

| Label | Support | GLiNER | Piiranha |
|---|---|---|---|
| PERSON | 7 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| ORG | 8 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| JOB | 2 | 0.50 / 1.00 / 0.83 | 0.00 / 0.00 / 0.00 |
| EMAIL_ADDRESS | 1 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 |
| LOCATION | 15 | 0.93 / 0.87 / 0.88 | 0.75 / 0.40 / 0.44 |
| AMOUNT | 2 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| DATE_TIME | 8 | 1.00 / 0.75 / 0.79 | 0.00 / 0.00 / 0.00 |
| UNIVERSITY | 1 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| PHONE_NUMBER | 1 | 0.50 / 1.00 / 0.83 | 1.00 / 1.00 / 1.00 |
| URL | 1 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |

Per label on `benchmark_extra` (P / R / F2):

| Label | Support | GLiNER | Piiranha |
|---|---|---|---|
| PERSON | 10 | 1.00 / 1.00 / 1.00 | 1.00 / 0.70 / 0.74 |
| ORG | 8 | 1.00 / 0.88 / 0.90 | 0.00 / 0.00 / 0.00 |
| JOB | 6 | 0.86 / 1.00 / 0.97 | 0.00 / 0.00 / 0.00 |
| EMAIL_ADDRESS | 4 | 1.00 / 1.00 / 1.00 | 1.00 / 0.25 / 0.29 |
| LOCATION | 6 | 0.86 / 1.00 / 0.97 | 0.71 / 0.83 / 0.81 |
| AMOUNT | 5 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| DATE_TIME | 13 | 0.91 / 0.77 / 0.79 | 0.00 / 0.00 / 0.00 |
| UNIVERSITY | 3 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| PHONE_NUMBER | 3 | 0.60 / 1.00 / 0.88 | 1.00 / 1.00 / 1.00 |
| URL | 3 | 1.00 / 0.67 / 0.71 | 0.00 / 0.00 / 0.00 |
| IBAN | 3 | 0.67 / 0.67 / 0.67 | 0.00 / 0.00 / 0.00 |
| SSN | 3 | 1.00 / 0.33 / 0.38 | 1.00 / 1.00 / 1.00 |
| USERNAME | 3 | 0.75 / 1.00 / 0.94 | 0.60 / 1.00 / 0.88 |
| PASSWORD | 2 | 0.67 / 1.00 / 0.91 | 1.00 / 0.50 / 0.56 |

Both benchmarks are tiny (support 1 to 15 per label), so per-label scores are anecdotal. `benchmark` also has labelling inconsistencies (see METHOD.md).

## Licences
- GLiNER `gliner_large-v2.5`: Apache-2.0.
- Piiranha: CC-BY-NC-ND 4.0 (attribution, non-commercial only, no redistributing modified weights), so this project is for benchmark and demo use only, not cleared for production.
