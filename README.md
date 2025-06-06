<h1 align="center"> Voice-Driven Mood Detection & Entertainment Recommendation</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-red" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/Model-CNN%20%2B%20Wav2Vec2-blue" alt="Model">
  <img src="https://img.shields.io/badge/Recommendation-System-green" alt="Recommendation">
</p>

<p align="center">
  <em>A full-stack AI system that listens to your voice, detects your mood, and recommends the perfect entertainment content 🎵🎬</em>
</p>

---

## Overview

This project integrates **emotion recognition** and **personalized content recommendation** into a seamless pipeline. We first detect emotional tone from user speech using a **Wav2Vec2 + CNN model**, then recommend content (e.g., music, videos) based on detected emotions.

---

## Features

- **Voice Emotion Detection**: Identify emotional tone (Happy, Sad, Angry, Neutral, Surprise)
- **Deep Learning Pipeline**: Wav2Vec2 feature extraction + CNN classifier
- **Mood-Based Recommendation**: Suggests YouTube or Spotify content based on emotion
- **Model Evaluation**: Uses CrossEntropy loss, accuracy metrics
- **Scalable Deployment Ready**: Future extensions include real-time apps or browser add-ons

---

## Dataset: Emotional Speech Data (ESD)

- **Link**: [GitHub - HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data)
- **Languages**: Multilingual (we used 3 English speakers)
- **Classes**: Neutral, Happy, Sad, Angry, Surprise
- **Format**: `.wav` audio clips with labeled emotions

> **Citation**: Kun Zhou et al., "Emotional voice conversion: Theory, databases and ESD", *Speech Communication*, 2022

---

## Methodology

### Audio Preprocessing
- Used Hugging Face’s **Wav2Vec2** to extract audio embeddings of shape `[batch, seq_len, 768]`
- Handled variable-length sequences via custom `collate_fn`
- Batched data with PyTorch's `DataLoader`

### Model Architecture: CNN on Wav2Vec2

```text
[768 features] -
→ Conv1D(256) - BatchNorm - ReLU - Pool -
→ Conv1D(128) - BatchNorm - ReLU - Pool -
→ Conv1D(64) - BatchNorm → ReLU - Pool -
→ Flatten - FC(128) - Dropout - FC(5 classes)
```

- **Loss**: `CrossEntropyLoss`
- **Optimizer**: `SGD`
- **Classes**: 5 (Neutral, Happy, Sad, Angry, Surprise)

### Training Strategy

- **Train/Val/Test Split**: 70% / 15% / 15%
- Monitored loss and accuracy on validation set
- Final model ready for real-time predictions

---

## Results

| Metric   | Value     |
|----------|-----------|
| Accuracy | ~85–90%   |
| Classes  | 5         |

- Validation results show strong generalization despite speaker variation
- Emotion classification performance varies across categories (e.g., Sad vs. Surprise)

---

## Recommendation System

Once emotion is detected, the system maps it to a type of content:

| Emotion  | Recommendation                    |
|----------|------------------------------------|
| Happy    | Upbeat songs / comedy videos    |
| Sad      | Comforting music / calm videos  |
| Angry    | Meditation / nature documentaries |
| Surprise | Curious facts / light humor     |
| Neutral  | Top charts / trending content   |

Future integrations could link real-time results to:
- **YouTube API**
- **Spotify API**
- **Tidal, Netflix**, etc.
---

## Future Work

- Web app or mobile app integration (e.g., Streamlit or Flask)
- Add multilingual emotion detection support
- Real-time streaming audio analysis

---

<p align="center">
  <i>“Your voice tells a story — let AI interpret it and entertain you.”</i>
</p>
