"""Streamlit UI for parsing and extracting recommendations from comment CSV files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from parser import extract_recommendations, parse_comments_csv

st.set_page_config(page_title="Comment CSV Parser", page_icon="🗂️", layout="centered")
st.title("Comment CSV Parser")

tab_csv, tab_paste = st.tabs(["Upload CSV", "Paste Text"])


def _run_extraction(input_df: pd.DataFrame, key_suffix: str) -> None:
    st.subheader("Options")
    min_confidence = st.slider(
        "Minimum confidence",
        min_value=-5,
        max_value=8,
        value=2,
        step=1,
        help="Only keep recommendations at or above this score.",
        key=f"min_confidence_{key_suffix}",
    )
    st.caption("Strict mode is always on: only clear Title + Author matches are returned.")

    if st.button("Run extraction", type="primary", key=f"run_{key_suffix}"):
        output_df = extract_recommendations(
            input_df,
            aggressive_mode=False,
            include_missing_author=False,
            min_confidence=min_confidence,
            include_metadata=True,
            drop_invalid_author=True,
        )
        output_df = output_df.reset_index(drop=True)

        st.subheader("Output Preview")
        if output_df.empty:
            st.info("No recommendations found for the selected options.")
        else:
            st.dataframe(output_df.head(200), use_container_width=True)
            st.caption(f"Extracted {len(output_df)} recommendation rows.")

        st.download_button(
            label="Download extracted CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="extracted_recommendations.csv",
            mime="text/csv",
            key=f"download_{key_suffix}",
        )


with tab_csv:
    st.write("Upload a CSV with columns `display_name` and `message`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            comments = parse_comments_csv(uploaded_file)

            if not comments:
                st.warning("No comment rows were found in the CSV.")
            else:
                input_df = pd.DataFrame(comments)
                st.success(f"Loaded {len(input_df)} comments successfully.")
                st.subheader("Input Preview")
                st.dataframe(input_df.head(50), use_container_width=True)
                _run_extraction(input_df, key_suffix="csv")

        except ValueError as err:
            st.error(str(err))
        except Exception:
            st.error("Failed to parse the file. Please upload a valid CSV.")


with tab_paste:
    st.write("Paste any text below — one comment or recommendation per line works best.")
    pasted = st.text_area(
        "Paste text here",
        height=300,
        placeholder="e.g.\nThe Name of the Wind by Patrick Rothfuss\nProject Hail Mary - Andy Weir",
        key="paste_input",
    )

    if pasted and pasted.strip():
        lines = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        input_df = pd.DataFrame({"display_name": ["pasted"] * len(lines), "message": lines})
        st.success(f"Loaded {len(lines)} line(s) of text.")
        st.subheader("Input Preview")
        st.dataframe(input_df.head(50), use_container_width=True)
        _run_extraction(input_df, key_suffix="paste")
