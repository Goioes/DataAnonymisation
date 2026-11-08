# Method

How the anonymizer works and how it is scored.

## Pipeline

```
text ──► detector (one per model) ──► entities ──► anonymize ──► text with <LABEL> placeholders
```

A detector wraps one pre-trained model and returns entities (text, label, start, end) in our label schema. `anonymize` replaces each entity with `<LABEL>`, working right to left so earlier offsets stay valid. An entity that overlaps one already replaced is skipped. The demo app, the API and the benchmark all use the same detectors, so the benchmark measures exactly what the demo runs.

## Labels

14 labels, one constant (`ALL_LABELS` in `detectors.py`):

- 12 required by the assignment: PERSON, ORG, JOB, EMAIL_ADDRESS, LOCATION, AMOUNT, DATE_TIME, UNIVERSITY, PHONE_NUMBER, URL, IBAN, SSN.
- 2 extra: USERNAME, PASSWORD. They are not in `benchmark.csv`.

All 14 are the scoring set on every benchmark. Predicted labels outside the 14 are discarded before scoring.

## Models

| Model | Type | Languages | Licence |
|---|---|---|---|
| GLiNER `gliner-community/gliner_large-v2.5` | Zero-shot span extraction; labels are passed as natural-language prompts at inference | English-focused checkpoint | Apache-2.0 |
| Piiranha `iiiorg/piiranha-v1-detect-personal-information` | Fine-tuned `mdeberta-v3-base` token classifier, 256-token context | EN, ES, FR, DE, IT, NL | CC-BY-NC-ND 4.0 |

**GLiNER** covers all 14 labels. Each of our labels is passed as a prompt phrase (`person`, `organization`, `job title`, `bank account` for IBAN, and so on, see `GlinerDetector.LABELS`). The confidence threshold is fixed at 0.75, a common default, and deliberately not tuned: the benchmarks are too small to tune on without flattering the scores. Prompt wording is not tuned either. Zero-shot spans depend on which labels are in the prompt, so the extra labels change what GLiNER finds for the others (for example it can tag an email address as USERNAME instead of EMAIL_ADDRESS). That is reported as a result, not tuned away.

**Piiranha** has no ORG, JOB, AMOUNT, UNIVERSITY or URL label, so these score 0 by construction. It has no country or state label either, so LOCATION recall is limited. Its source labels are mapped as follows; anything not listed is dropped (CREDITCARDNUMBER, DRIVERLICENSENUM have no target label):

| Piiranha | Ours |
|---|---|
| GIVENNAME, SURNAME | PERSON (adjacent spans are merged) |
| CITY, STREET, BUILDINGNUM, ZIPCODE | LOCATION |
| TELEPHONENUM | PHONE_NUMBER |
| SOCIALNUM, IDCARDNUM, TAXNUM | SSN |
| EMAIL | EMAIL_ADDRESS |
| ACCOUNTNUM | IBAN (approximate) |
| DATEOFBIRTH | DATE_TIME (partial: the benchmarks use years and dates, not birth dates) |
| USERNAME, PASSWORD | USERNAME, PASSWORD |

IDCARDNUM and TAXNUM to SSN is an untested guess (not yet checked with a Belgian national number). The model card gives no guidance on merging tokens, so the wrapper merges adjacent same-label tokens, and trims leading whitespace and trailing punctuation that the tokens carry.

Models are downloaded from Hugging Face at runtime (not committed) and loaded once, at startup. Each is pinned to a commit hash (`REVISION` in its detector class), so the weights cannot change under the benchmark results.

## Evaluation

Entity-level matching with Microsoft's `presidio-evaluator` (`SpanEvaluator`), used as a library. Pinned via `uv.lock` because the library is pre-1.0 and its API changes.

1. **Gold spans.** Each benchmark row has the plain text and the same text with each entity replaced by its `<LABEL>`. The gold spans are recovered by aligning the two: literal text is kept, each placeholder becomes a capture group. Every row of every selected benchmark is validated before any model loads. A row that does not align, or uses a label outside the 14, stops the run with file, row number and reason.
2. **Tokens.** Text is tokenized with a spaCy blank tokenizer (no model download, no language-specific rules). Gold spans and predictions are turned into per-token tags, the format the evaluator expects.
3. **Matching.** `SpanEvaluator` with character-level IoU threshold 0.5 (set explicitly: the library's code default is 0.9 while its docstring says 0.5). The evaluator merges adjacent same-type spans separated by skip words, so an address split on commas into several LOCATION labels still matches as one. Several same-type predictions overlapping one gold span are scored by their cumulative IoU.
4. **Wrong label.** The right span with the wrong label counts as a false negative for the gold label plus a false positive for the predicted label.

**Skip list.** Whitespace and punctuation only (`SKIP_WORDS` in `evaluate.py`), no words. The Presidio default list contains words such as `and`, `street`, `de`, `first`, which has two problems: "Bill Gates and Paul Allen" (two gold spans) would merge into one, and stripping English and Dutch/French words makes the matching lenient in a different way per language.

**Metrics.**
- Precision, recall, F1 and F2. The headline is **F2**: recall is weighted double, because a missed entity is a privacy leak while over-redaction only costs utility. The Presidio docs recommend the same.
- Micro aggregation: per-label true positives, false positives and false negatives are summed over all labels.
- A global type-agnostic score ("PII vs O", ignores the label) is reported next to it. It is the closest thing to a leak rate.
- A label with gold support 0 is shown as n/a. A label a model does not support scores 0, and the reason is the mapping above.

Results are written to `results/<benchmark>/<model>.json` (per-label table, micro, global, error list, per-example spans), overwriting that pair only.

## Benchmarks

**`benchmark.csv`** (10 examples, English, given with the assignment). Known problems; labels are reported as-is, not corrected:
- Inconsistent labelling: "CEO" is JOB but "President" and "co-founder" are not; the period after "Apple Inc." is in or out of the label depending on the row.
- Debatable labels: "area code 212" is LOCATION, "US citizen" is LOCATION.
- Addresses are split on commas into several LOCATION labels.
- Landmarks (Eiffel Tower) are unlabelled, so models that tag them get false positives.
- No IBAN or SSN examples; EMAIL, URL, PHONE and UNIVERSITY have support 1, so per-label scores on these are anecdotal.
- All entities are famous (memorization risk), all text is encyclopedic English prose.

**`benchmark_extra.csv`** (our own). Generated by AI (Claude Sonnet 5.5, medium effort), then reviewed by a human. Multilingual (French, Dutch, English) and covers all 14 labels, including IBAN, SSN, USERNAME and PASSWORD. Same `text;label` format plus a `language` column that is passed through to the error list. This is our reading of the assignment's "Extra": an own labelled file with more entities.

## Adding a benchmark or a model

- **Benchmark:** drop a `text;label` CSV (semicolon-separated) in `Data/`. The file name is its name. `evaluate.py` and the Benchmark page find it by globbing; no code change.
- **Model:** write one class with `detect(text) -> list[Entity]` in `detectors.py`, keeping its label mapping inside the class, and add it to the `DETECTORS` registry. The demo, the benchmark and the API pick it up.

## Limitations

- No length handling: text beyond a model's token limit may be left unprocessed, so the "anonymized" output can still contain sensitive data. The API rejects texts over 1000 characters. Chunking inside each detector would fix this.
- No regex layer, so structured entities (IBAN, SSN, email) depend entirely on the models.
- The benchmarks are small: differences of a few entities move the scores.
- Not for production: Piiranha's licence is non-commercial and the models are not validated for real personal data.
