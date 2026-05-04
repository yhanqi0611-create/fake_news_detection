---
title: Fake News Detector
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
short_description: Flag likely fake vs real news from text using embeddings + sklearn.
data_source: 
    https://www.kaggle.com/datasets/nitishjolly/news-detection-fake-or-real-dataset
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
hugguing_face_dataset:
    https://huggingface.co/datasets/yahuqiao/fake-real-news
working_demo:
    https://huggingface.co/spaces/yahuqiao/fake-news-detector
    
---

```markdown
## How to Run

Follow these steps to set up the environment and run the Fake News Detector locally.

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.

### 2. Install Dependencies
This project requires specific library versions to ensure compatibility (especially for `numpy` and `scikit-learn`). Run the following command:

```bash
pip install -r requirements.txt
```

### 3. Run the Training Script
If you want to re-train the model or explore the embeddings:

```bash
python train_model.py