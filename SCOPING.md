# Scoping: Data Anonymization (NLP exercise)

Status: draft. Open points are listed per step in section 12 and in the risks of section 7.

## 1. Goal
Extract sensitive entities from text with pre-trained NER models, anonymize them, and benchmark 2 models against the labelled dataset. Source: `instructions.docx`, `Data/benchmark.csv`.

## 2. Scope

| | Item |
|---|---|
| **Must** | Anonymization pipeline (2 models), entity-level benchmark on `benchmark.csv`, reproducible env, README, presentation, demo (Streamlit app: demo page + benchmark page) |
| **Should** | `POST /anonymize` API (optional in the assignment), standalone |
| **Nice to have** | Second benchmark: 10 AI-generated multilingual examples; chunking of long texts inside each detector (sentence/paragraph boundaries, offsets mapped back to the full text) |
| **Out** | Regex layer, auth, database, Docker, custom frontend, fine-tuning |

## 3. Entities (12 labels from the assignment)
PERSON, ORG, JOB, EMAIL_ADDRESS, LOCATION, AMOUNT, DATE_TIME, UNIVERSITY, PHONE_NUMBER, URL, IBAN, SSN.
Anonymized output uses `<LABEL>` placeholders, same as the dataset.

## 4. Models
| Model | Type | Notes |
|---|---|---|
| GLiNER (`gliner-community/gliner_large-v2.5`) | Zero-shot span extraction, labels passed at inference | Covers all 12 labels; label wording and threshold need tuning; token limit (chunking is a nice-to-have, see section 2). Chosen over the multilingual checkpoint for English accuracy on `benchmark.csv`; its behaviour on NL/FR/DE/TR is a result of the second benchmark, not a claim (verify the language coverage on the model card). |
| Piiranha (`iiiorg/piiranha-v1-detect-personal-information`) | Fine-tune of `mdeberta-v3-base` (token classifier), 6 languages (EN, ES, FR, DE, IT, NL), 256-token context | No ORG, JOB, AMOUNT, UNIVERSITY, URL; reported as unsupported. No country or state label either, so LOCATION recall is structurally limited (see below). Model card gives no aggregation guidance: adjacent tokens are merged in our wrapper. |

Each model is loaded on first use and cached (not at startup), so startup time does not grow with every detector added to the registry. Weights are downloaded from Hugging Face at runtime, not committed.

Piiranha label mapping (checked against the model card, 17 PII labels):

| Piiranha | Ours |
|---|---|
| GIVENNAME, SURNAME (merged when adjacent) | PERSON |
| CITY, STREET, BUILDINGNUM, ZIPCODE | LOCATION (partial: no country or state label) |
| TELEPHONENUM | PHONE_NUMBER |
| SOCIALNUM | SSN |
| EMAIL | EMAIL_ADDRESS |
| ACCOUNTNUM | IBAN (approximate, no evidence on the card) |
| DATEOFBIRTH | DATE_TIME (partial; the dataset has years, not birth dates, so expect near 0) |
| IDCARDNUM, TAXNUM | SSN candidates; decide in step 1 by testing a Belgian national number |
| CREDITCARDNUMBER, PASSWORD, USERNAME, DRIVERLICENSENUM | Dropped (no target label) |

Piiranha's weakest reported label is SURNAME (recall 0.78); the card says first/last names are often confused but still flagged as PII, which is why GIVENNAME and SURNAME map to the same label.

## 5. Architecture

```
          ┌─► app.py (Streamlit: demo page + benchmark page)
detectors ┤
anonymize ┴─► api.py (FastAPI, optional, standalone)
evaluate ──► results file ──► benchmark page + README table
```

- One flat package, one file per part: `detectors.py`, `anonymize.py`, `evaluate.py`, `app.py`, `api.py`. Split into folders only when a file gets painful.
- `Entity(text, label, start, end)` is the only shared type.
- Detectors: one class per model with `detect(text) -> list[Entity]`. Plain duck typing, no base class (two models share no logic). A dict registry `{name: detector}` feeds the dropdown and `GET /models`.
- **Adding a detector:** write one class with `detect(text) -> list[Entity]` (its label mapping stays inside the class) and add one line to the registry. The demo, benchmark script and API pick it up with no other change; the benchmark page shows every model found in the result file.
- Dependencies point one way: `app` and `api` use `detectors` and `anonymize`; `evaluate` uses `detectors` only; detectors know nothing about them. The benchmark therefore measures exactly what the demo runs.
- Models are loaded lazily on first selection/request and cached per name (`st.cache_resource` on a `get_detector(name)` function in Streamlit, a cached lookup in the API).

## 6. API (FastAPI, optional, standalone)
- `POST /anonymize` with `{text, model}` returns `{anonymized_text, entities: [{text, label, start, end}]}`
- `GET /models` returns the available models
- Not used by the Streamlit app. It is the entry point for the assignment's optional predict endpoint; `/docs` works as a second demo.
- Why FastAPI: typed request/response validation (pydantic), automatic OpenAPI docs, minimal boilerplate. Async and raw speed are not reasons (inference is CPU-bound).

## 6b. App (Streamlit, imports the detectors directly)
- **Demo page:** text input (pre-filled from a dropdown of `benchmark.csv` rows, still editable), model dropdown, anonymized text, entity table (text, label).
- **Benchmark page:** per-model, per-label results. Reads the result file written by the benchmark script.
- One process, one command. Does not depend on the API.

## 7. Evaluation

**Matching (entity level): `presidio-evaluator` `SpanEvaluator`, used at runtime**
1. Gold spans are recovered by aligning the original text with the labelled text (literal text kept, each `<X>` becomes a capture group). Own code.
2. Conversion layer (own code): tokenize each text with a spaCy blank tokenizer (no model download), then turn gold spans and model predictions into per-token tags, the format the evaluator expects.
3. `SpanEvaluator(iou_threshold=0.5, char_based=True, ...)`, `beta=2`, skip-word list passed explicitly via `skip_words` (open decision, see "Reference and validation"). The evaluator normalizes tokens, merges adjacent same-type spans separated by skip words/punctuation (removes the address-splitting convention), and matches on character-level IoU: shared characters / characters covered by either span. The threshold is set explicitly because the code default (0.9) and its docstring (0.5) disagree.
4. Right span with wrong label = FN for the gold label plus FP for the predicted label. Results come from `calculate_score_on_df(level="both")`: per-type counts and P/R/F2, plus a global type-agnostic "PII vs O" score.

**Metrics**
- Precision, recall, F1 everywhere; **headline = F2** (recall weighted double; a missed entity is a privacy leak, over-redaction is only a utility loss). The Presidio docs recommend the same: "we recommend to use the beta=2 score, which gives more importance to recall" (https://presidio.dataprivacystack.org/evaluation/).
- Aggregation: micro only, computed by summing the evaluator's per-type TP/FP/FN. The evaluator's global type-agnostic score is shown as well (closest thing to a leak rate).
- Per-label table: support, P, R, F1, F2 per model. Unsupported label shown as 0 with a flag; support 0 shown as n/a.

**Reference and validation**
- Chosen because it is Microsoft's open-source reference tooling (presidio-research), covers normalization, merging, char-level IoU, F-beta and per-type results out of the box, and has a configurable skip-word list.
- Risks and the result of the step 0 spike (installed `presidio-evaluator` 0.3.x, `transformers` 5.22.2):
  - Pre-1.0; the API is already changing (`calculate_score()` was removed). Pin the version via `uv.lock`.
  - Heavy dependencies: **resolved**. GLiNER, `SpanEvaluator` and Piiranha import and run in one `uv` environment.
  - Multiple overlaps: **checked, acceptable**. Same-type predictions overlapping one gold span are scored by their cumulative IoU (e.g. gold "New York City" vs predictions "New York" + "City" = 1 TP).
  - Tokenizer-dependent boundaries: still to test in step 3 on the dataset's edge cases (`Apple Inc.`, comma-separated addresses). `inc.` is a skip word, so the trailing period of "Apple Inc." does not matter.
  - Span merging: **confirmed**. The default skip-word list contains `and`, so "Bill Gates and Paul Allen" (two gold PERSON spans) merges into one span (1 TP instead of 2; a model finding only "Bill Gates" gets 1 FP + 1 FN instead of 1 TP + 1 FN). The `skip_words` argument replaces the default list, verified by removing `and`.
  - **Open decision, skip list:** recommended is the default Presidio list (`get_skip_words()`) minus `and`. The default also strips words like `street`, `de`, `first`, `last`, `year`, `old`, `day`, `inc.` before matching, which is lenient ("Main Street" matches "Main"). Alternative: a short hand-picked list (punctuation, whitespace, `inc.`), stricter but more work. Decide in step 3 and write the final choice in the README.
  - `iou_threshold`: **confirmed**, the code default is 0.9 and the docstring says 0.5. We set 0.5 explicitly. The beta=2 recommendation is taken from the Presidio docs as quoted above.
- Fallback if the evaluator causes problems later: own implementation of the same semantics (about 40 lines).

**Known dataset problems (labels reported as-is, not corrected; the evaluator's span merging is the only normalization applied, which softens the comma-split addresses)**
- Inconsistent labelling: "CEO" is JOB but "President" and "co-founder" are not; the period after "Apple Inc." is in/out of the label depending on the row.
- Wrong/debatable labels: "area code 212" = LOCATION; "US citizen" = LOCATION.
- Addresses are split on commas into several LOCATION labels.
- Landmarks (Eiffel Tower) are unlabelled, so models that tag them get false positives.
- No IBAN or SSN examples; EMAIL, URL, PHONE, UNIVERSITY have support 1. Per-label scores on these are anecdotal.
- All entities are famous (memorization risk), all English, encyclopedic prose.

## 8. Second benchmark (nice to have)
- 10 AI-generated, **human-reviewed** examples, same `<LABEL>` format.
- Languages: NL, FR, DE, plus Turkish (outside Piiranha's 6 languages) and others to be chosen. The chosen GLiNER checkpoint is English-focused, so this benchmark tests both models outside their comfort zone in different ways.
- Must include IBAN and SSN.
- Fictitious data only: made-up names, test IBANs, invalid SSNs. No real personal data.
- About one example per language, so no per-language scores; show a per-example table (language, gold, found, missed, extra).
- This is our reading of the assignment's "Extra" (own labelled file for benchmarking). The assignment literally asks for more entities; IBAN and SSN are already among the 12, so state this interpretation in the presentation.

## 9. Licences (README section)
- Piiranha: CC-BY-NC-ND 4.0 (confirmed on the model card; the card does not summarise the terms, so the wording below comes from the CC licence). Attribution required, non-commercial only, no redistributing modified weights. This project: benchmark and demo only, not cleared for production.
- GLiNER `gliner_large-v2.5`: Apache-2.0.

## 10. Environment and delivery
- `uv` + `pyproject.toml` + lockfile, Python >= 3.12 (already in repo).
- Deliverables: GitHub repo, README, presentation, demo.
- Way of working to explain in the presentation (general procedure, not a command log):
  1. Scope first: decisions and trade-offs written down before code (this file).
  2. Check the riskiest assumption early (dependencies resolve together).
  3. Build in small vertical slices (models, anonymize, benchmark, UI, API), each runnable on its own.
  4. Version control: work directly on `main`, one commit per working slice after human review. `main` always runs.
  5. AI-assisted code and AI-generated data are a starting point: reviewed by a human before they are used or published.
- `instructions.docx` (the assignment brief) is gitignored, as are `.vscode/` and `.pytest_cache/`. `Data/benchmark.csv`, `uv.lock`, `.python-version` and the benchmark result file stay tracked.


## 11. Decisions so far
- Demo: Streamlit, imports the detectors directly (one process). API: FastAPI, optional and standalone, `POST /anonymize`, `GET /models`.
- Architecture: flat package, shared `Entity` type, duck-typed detectors, no base class (section 5).
- Models: GLiNER (`gliner_large-v2.5`, English-focused, chosen for accuracy on the English benchmark) and Piiranha. No regex layer.
- Extensibility: a new detector is one class plus one registry line (section 5).
- Model loading: lazy on first use, cached per model name (sections 4 and 5).
- Demo page opens pre-filled with a benchmark row, with a dropdown to pick other rows; the text stays editable.
- GLiNER threshold is a fixed constant inside the detector, tuned once in step 1. No slider in the demo, so the demo and the benchmark run the same thing. Threshold sensitivity stays parked (section 13).
- Chunking of long texts belongs inside each detector (not in callers), built as a nice-to-have.
- Until chunking exists, detectors do no length handling (demo scope). Known limitation, to state in the README: text beyond a model's token limit may be left unprocessed, so the "anonymized" output can still contain sensitive data. The API's planned length limit stays.
- GLiNER `gliner_large-v2.5` licence: Apache-2.0 (confirmed).
- Build order: steps 1 and 2, then the demo page, then step 3, then the benchmark page.
- Matching: `presidio-evaluator` `SpanEvaluator`, IoU threshold 0.5 (a parameter; explained in the presentation), beta = 2, micro aggregation plus the global type-agnostic score. Skip list still open (section 7); the default list merges spans over "and".
- Step 0 spike passed: imports and Piiranha load work in one environment, evaluator semantics checked (section 7).
- Benchmark page reads the result file written by the offline benchmark script (not served via the API).
- Second (multilingual) benchmark is nice to have.

## 12. Plan

Each step ends with a check; do not start the next step before it passes.

| Step | Work | Done when |
|---|---|---|
| 0. Setup | Dependencies are already in `pyproject.toml` and `uv.lock` (gliner, transformers, torch, presidio-evaluator, fastapi, uvicorn, streamlit, pandas). Import GLiNER and `SpanEvaluator` in one environment, load the Piiranha pipeline once, and run a `SpanEvaluator` smoke test on a tiny example (IoU default, skip-word list, multiple-overlap behaviour). Fill in README skeleton and the `pyproject` description. | `uv sync` works from a clean checkout, both imports succeed, Piiranha loads, smoke test behaves as documented. If it fails, the matching falls back to our own implementation of the same semantics. |
| 1. Models | Wrappers for GLiNER and Piiranha that return character spans in our 12-label schema. Piiranha mapping and licence are checked on the model card; GLiNER licence still to verify. Decide IDCARDNUM/TAXNUM to SSN by testing. GLiNER checkpoint is decided (`gliner_large-v2.5`); tune the threshold. Piiranha observations from step 0 (`try_piiranha.py`): the model returns `I-` tags only; a lone `▁` token starts some entities with the span beginning at the preceding space, and some tokens carry a trailing comma or period, so the wrapper trims whitespace and trailing punctuation before merging. The tokenizer was checked (the `transformers` tokenizer and the repo's `spm.model` split identically, so it is not a tokenizer bug). On short sentences Piiranha confidently labels names (`Tim Cook`, `Jan Peeters`), an email and an IBAN as `O`, while finding cities, streets, zip codes and phone numbers; test on a benchmark-style paragraph, and if names are still missed report it as a result. If it looks like a library effect, retest under `transformers` 4.x in a throwaway environment. Windows console needs `PYTHONUTF8=1` to print the `▁` character. | Both models run on one sentence from the command line. |
| 2. Anonymize | `anonymize(text, spans)` produces `<LABEL>` text. | Both models anonymize one sentence from the command line. |
| 3. Benchmark | Recover gold spans from `benchmark.csv`. Conversion layer to per-token tags. `SpanEvaluator` with threshold 0.5 and beta 2. Script writes a result file (per-label table, micro, global score, error list). One small test file for gold alignment and `anonymize`. | All 10 rows align; result file written; errors inspected by hand for plausibility. |
| 4. App | Streamlit demo page (imports the detectors) and benchmark page (reads the result file). | Both pages work end to end. |
| 5. Delivery | README (setup, usage, endpoints, method, label mapping, dataset caveats, licences). Presentation (way of working, decisions, results, limits, future work). | A reviewer can run it from the README alone. |
| Should: API | FastAPI endpoints (section 6) with input validation (empty text, unknown model, length limit). Can be built any time after step 2. | `POST /anonymize` works for both models. |
| Nice to have | Multilingual benchmark (section 8), inserted after step 3 if time allows. Human review of the examples before use. | Second result file shown on the benchmark page. |

Open choices to settle during the steps: GLiNER threshold (step 1), skip list (step 3). Project layout is decided: flat package (section 5).

## 13. Parked ideas (not in scope)
- Confusion matrix heatmap (the evaluator can produce it).
- Latency per example.
- IoU threshold sensitivity (extra line at 0.75).
- Regex layer for structured entities. IBAN and SSN depend entirely on the models for now, likely weak for IBAN.

## 14. Status and next steps (handoff)

Last updated after the first build session. Work happens directly on `main`; changes are committed by the author after checks.

**Done**
- Step 0 (setup and spike), step 1 (`detectors.py`: `Entity`, `GlinerDetector`, `PiiranhaDetector`, `DETECTORS` registry, cached `get_detector`), step 2 (`anonymize.py`), demo page (`app.py`, checked headlessly with Streamlit `AppTest`, not yet in a browser).
- Terms are in `CONTEXT.md`.

**Open in step 1**
- IDCARDNUM / TAXNUM to SSN: test with a Belgian national number.
- Optional retest of Piiranha under `transformers` 4.x in a throwaway environment.

**Next: step 3 (benchmark), then the benchmark page**
- Open decisions: skip list for `SpanEvaluator` (section 7) and the result-file format (the contract between `evaluate.py` and the benchmark page; JSON or CSV, one file per benchmark or combined).
- `Data/benchmark.csv` is semicolon-separated (`text;label`): read it with `sep=";"`.
- One small test file for gold alignment and `anonymize` (section 12).

**Observations to keep**
- Piiranha misses person names (`Elon Musk`, `Tim Cook`, `John Smith`) even in benchmark-style text, and returns only part of an email (`tim@`), so `anonymize` can leave `apple.com` in the text. Report as a result.
- GLiNER finds almost all entities on the sample rows; `US` in "US citizen" is tagged LOCATION, which matches the dataset's own debatable label.
- GLiNER's prompt wording for IBAN is `"bank account"`, mapped to the label `IBAN`. The label schema stays at the 12 labels.

**Practical**
- `try_gliner.py` and `try_piiranha.py` are kept on purpose for manual testing; their logic is duplicated in `detectors.py`.
- On Windows run Python with `PYTHONUTF8=1`.
