"""Streamlit demo: anonymize a text with the chosen model. Run: uv run streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from anonymize import anonymize
from detectors import DETECTORS, get_detector


@st.cache_data
def load_examples():
    return pd.read_csv(Path(__file__).parent / "Data" / "benchmark.csv", sep=";")["text"].tolist()


examples = load_examples()

st.title("Data anonymization")

row = st.selectbox("Example", range(len(examples)), format_func=lambda i: f"{i + 1}. {examples[i][:70]}...")
# keyed per example, so picking another example replaces the text
text = st.text_area("Text", examples[row], key=f"text_{row}", height=150)
model = st.selectbox("Model", list(DETECTORS))

if st.button("Anonymize"):
    with st.spinner(f"Running {model} (the first run loads the model)..."):
        entities = get_detector(model).detect(text)
    st.subheader("Anonymized text")
    st.text(anonymize(text, entities))  # st.text: markdown would swallow the <LABEL> placeholders
    st.subheader("Entities")
    if entities:
        st.dataframe(pd.DataFrame([{"text": e.text, "label": e.label} for e in entities]), hide_index=True)
    else:
        st.info("No entities found.")
