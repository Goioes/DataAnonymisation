"""Benchmark page: reads results/<benchmark>/<model>.json written by evaluate.py. Knows no benchmark by name."""
import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from anonymize import anonymize
from detectors import Entity
from evaluate import IOU_THRESHOLD

EXTRA = {"USERNAME", "PASSWORD"}  # labels beyond the assignment (SCOPING.md section 3)
files = sorted((Path(__file__).resolve().parent.parent / "results").glob("*/*.json"))

st.title("Benchmark")
if not files:
    st.info("No results yet. Run `uv run python evaluate.py`.")
    st.stop()

name = st.selectbox("Benchmark", sorted({f.parent.name for f in files}))
results = [json.loads(f.read_text(encoding="utf-8")) for f in files if f.parent.name == name]

st.caption(f"{results[0]['examples']} examples · scored at " + " · ".join(f"{r['model']}: {r['scored_at']}" for r in results))
st.caption("**P** = correct when flagged · **R** = share of PII found · **F1** = balance of P and R · **F2** = same, but missing PII counts more.")


def overall(key):
    return pd.DataFrame([{
        "model": r["model"],
        **{h: round(r[key][k], 2) for k, h in [("precision", "P"), ("recall", "R"), ("f1", "F1"), ("f2", "F2")]},
    } for r in results])


st.subheader("Detection (any PII, label ignored)")
st.caption("Was the PII found at all? A name tagged as \"address\" still counts, because it would be redacted anyway. "
           "Recall (R) is the one to watch: 1 − R is the share that leaks.")
st.dataframe(overall("global"), hide_index=True)

st.subheader("Detection + correct label (strict)")
st.caption("Was the PII found and given the right label? The gap with the table above is the share of errors that are wrong labels.")
st.dataframe(overall("micro"), hide_index=True)

st.subheader("Per label")
st.caption("Support = number of gold spans with this label. n/a = no examples with this label in the benchmark.")
rows = []
for label in results[0]["labels"]:
    row = {"label": label + (" (extra)" if label in EXTRA else ""), "support": results[0]["labels"][label]["support"]}
    for r in results:
        s = r["labels"][label]
        for k, h in [("precision", "P"), ("recall", "R"), ("f1", "F1")]:
            row[f"{r['model']} {h}"] = f"{s[k]:.2f}" if s["support"] else "n/a"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---- Example browser: what was masked and what not -------------------------------------------------------------
COLORS = {"gold": "100,149,237", "correct": "46,160,67", "wrong label": "240,136,62", "leaked": "248,81,73", "false alarm": "170,100,220"}


def iou(a, b):
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    return inter / (a.end - a.start + b.end - b.start - inter)


def classify(gold, predicted):
    """(span, kind, badge) per gold span, plus false alarms. Same matching rule as the scoring (IoU threshold)."""
    out, matched = [], set()
    for g in gold:
        hit = max(predicted, key=lambda p: iou(g, p), default=None)
        if hit and iou(g, hit) >= IOU_THRESHOLD:
            matched.add(id(hit))
            out.append((g, "correct", g.label) if hit.label == g.label else (g, "wrong label", f"{g.label} → {hit.label}"))
        else:
            out.append((g, "leaked", f"{g.label}: leaked"))
    out += [(p, "false alarm", f"{p.label}: false alarm") for p in predicted if id(p) not in matched]
    return out


def highlight(text, marks):
    """HTML of `text` with marked spans; a span overlapping an earlier one is skipped."""
    html, pos = "", 0
    for span, kind, badge in sorted(marks, key=lambda m: m[0].start):
        if span.start < pos:
            continue
        html += escape(text[pos:span.start])
        html += (f'<span style="background:rgba({COLORS[kind]},.3);border-radius:4px;padding:0 3px">{escape(text[span.start:span.end])}'
                 f' <small style="opacity:.75">[{escape(badge)}]</small></span>')
        pos = span.end
    return f'<div style="line-height:1.9">{html + escape(text[pos:])}</div>'


st.subheader("Example browser")
if not all("details" in r for r in results):
    st.info("These results predate the example browser. Re-run `uv run python evaluate.py` to add it.")
    st.stop()
st.caption("Highlight colours: " + " · ".join(
    f'<span style="background:rgba({COLORS[k]},.3);padding:0 4px;border-radius:4px">{k}</span>' for k in list(COLORS)[1:]
) + ". Wrong label only counts as an error in the strict detection table.", unsafe_allow_html=True)

texts = [d["text"] for d in results[0]["details"]]
i = st.selectbox("Example", range(len(texts)), format_func=lambda i: f"{i + 1}. {texts[i][:70]}...")
gold = [Entity(**e) for e in results[0]["details"][i]["gold"]]
with st.container(border=True):  # bordered boxes follow the light/dark theme, unlike a hard-coded background
    st.markdown("##### Gold: what should be masked")
    st.markdown(highlight(texts[i], [(g, "gold", g.label) for g in gold]), unsafe_allow_html=True)

for col, r in zip(st.columns(len(results)), results):
    predicted = [Entity(**e) for e in r["details"][i]["predicted"]]
    marks = classify(gold, predicted)
    count = lambda k: sum(1 for _, kind, _ in marks if kind == k)
    with col, st.container(border=True):
        st.markdown(f"##### {r['model']}")
        st.markdown(f"**{count('correct') + count('wrong label')} of {len(gold)} PII masked** · {count('leaked')} leaked · "
                    f"{count('wrong label')} wrong label · {count('false alarm')} false alarm")
        st.markdown(highlight(texts[i], marks), unsafe_allow_html=True)
        st.divider()
        st.caption("Masked output")
        st.text(anonymize(texts[i], predicted))  # st.text: markdown would swallow the <LABEL> placeholders
