"""Public Streamlit demo for the production AI-impact classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ai_impact_classifier.production import predict_with_review_flags


MODEL_PATH = Path("models/e0_e1_e23_sparse_svc.joblib")
REQUIRED_BATCH_COLUMN = "keytask_content"


@st.cache_resource(show_spinner="Loading classifier...")
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def classify(frame: pd.DataFrame, *, diagnostics: bool = False) -> pd.DataFrame:
    artifact = load_artifact()
    results = predict_with_review_flags(
        artifact["model"], frame, artifact["review_calibration"], include_diagnostics=diagnostics
    )
    return pd.DataFrame(results)


st.set_page_config(page_title="AI Impact Classifier", page_icon="AI", layout="centered")
st.title("AI Impact Classifier")
st.caption("Task-level E0 / E1 / E23 classification with a CPU-first review-priority signal.")

single_tab, batch_tab = st.tabs(["Single task", "Batch CSV"])

with single_tab:
    with st.form("single_prediction"):
        title = st.text_input("Job role title (optional)", placeholder="e.g. Financial analyst")
        task = st.text_area(
            "Key task content",
            placeholder="Describe the work activity to assess.",
            height=150,
        )
        submitted = st.form_submit_button("Classify task", type="primary")
    if submitted:
        if not task.strip():
            st.error("Key task content is required.")
        else:
            result = classify(pd.DataFrame([{"jobrole_title": title, "keytask_content": task}])).iloc[0]
            left, right = st.columns(2)
            left.metric("Predicted label", result.predicted_label)
            right.metric("Review priority", result.review_level.title())
            st.progress(float(result.review_score), text=f"Review score: {result.review_score:.0%}")
            if result.needs_review:
                st.warning("This task should be reviewed before relying on the label.")
            elif result.review_level == "medium":
                st.info("The label is usable, with some ambiguity or vocabulary shift to monitor.")
            else:
                st.success("The task is within the model's familiar, higher-confidence range.")

with batch_tab:
    uploaded = st.file_uploader("Upload CSV", type="csv", help="Requires keytask_content; jobrole_title is optional.")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        if REQUIRED_BATCH_COLUMN not in batch:
            st.error("The CSV must include a keytask_content column.")
        else:
            inputs = batch.copy()
            if "jobrole_title" not in inputs:
                inputs["jobrole_title"] = ""
            with st.spinner("Classifying tasks..."):
                predictions = classify(inputs[["jobrole_title", "keytask_content"]])
            output = pd.concat([batch.reset_index(drop=True), predictions], axis=1)
            st.dataframe(output[["predicted_label", "review_level", "review_score", "needs_review"]], use_container_width=True)
            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="ai_impact_predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption("E23 combines the original E2 and E3 labels. Review score is an operational priority signal, not a probability.")
