import os

import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Avoid Hugging Face tokenizers fork warnings / potential deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "models/best_model.joblib"

# Convention from training script: label=True -> 1, label=False -> 0
# Display labels as: True / Fake
IDX_TO_LABEL = {1: "True", 0: "Fake"}


# Load once at Space startup
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
clf = joblib.load(MODEL_PATH)


def predict(text: str):
    """Text -> normalized embedding -> classifier prediction -> label + probability."""
    if text is None or not str(text).strip():
        return "Please enter a piece of news text.", ""

    # Sentence-Transformers supports normalize_embeddings=True directly
    emb = embedder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    pred_idx = int(clf.predict(emb)[0])
    pred_label = IDX_TO_LABEL.get(pred_idx, str(pred_idx))

    prob_str = ""
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb)[0]
        # Find the column index for the predicted class in predict_proba output
        classes = list(getattr(clf, "classes_", []))
        if classes:
            try:
                col = classes.index(pred_idx)
                prob = float(proba[col])
                prob_str = f"{prob:.4f}"
            except ValueError:
                prob_str = f"{float(np.max(proba)):.4f}"
        else:
            prob_str = f"{float(np.max(proba)):.4f}"

    return pred_label, prob_str


def _interface_kwargs():
    # Gradio 6+ renamed Interface(allow_flagging=...) -> Interface(flagging_mode=...)
    major = int(gr.__version__.split(".", 1)[0])
    if major >= 6:
        return {"flagging_mode": "never"}
    return {"allow_flagging": "never"}


demo = gr.Interface(
    fn=predict,
    title="Fake News Detector",
    inputs=gr.Textbox(lines=8, label="Input Text"),
    outputs=[
        gr.Textbox(label="Prediction (True or Fake)"),
        gr.Textbox(label="Probability (if available)"),
    ],
    **_interface_kwargs(),
)


if __name__ == "__main__":
    port = int(os.getenv("SPACE_PORT") or os.getenv("PORT") or "7860")
    demo.launch(server_name="0.0.0.0", server_port=port)

