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
- **Languages**: Mandarin & English with multi-spearkers
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
Conv1D(256) - BatchNorm - ReLU - Pool -
Conv1D(128) - BatchNorm - ReLU - Pool -
Conv1D(64) - BatchNorm → ReLU - Pool -
Flatten - FC(128) - Dropout - FC(5 classes)
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

## Integrating Deep Learning with Hugging Face & OpenAI APIs

### Emotion Classification with CNN (Wav2Vec2)

The final output of our CNN model is a predicted class label:
```python
emotion_label = model(audio_tensor)  # Output: 0–4 (Neutral, Happy, Sad, Angry, Surprise)
```

This prediction serves as the **input condition** for downstream recommendation.
---

### Connect with Hugging Face Transformers

You can integrate additional emotion understanding or text generation via Hugging Face. For example:

```python
from transformers import pipeline

emotion_to_prompt = {
    1: "Suggest happy songs or movies for someone feeling joyful",
    2: "Recommend comforting content for someone who's feeling sad",
    3: "Help a person calm down with soothing content",
    4: "Suggest something surprising and fun",
    0: "Recommend trending neutral entertainment"
}

# Use HF model for creative prompt generation
generator = pipeline("text-generation", model="gpt2")
prompt = emotion_to_prompt[emotion_label]
response = generator(prompt, max_length=30, do_sample=True)[0]['generated_text']
```

---
### OpenAI GPT for Recommendation
Use the predicted emotion to query OpenAI GPT to return entertainment suggestions in real time:

```python
import openai

openai.api_key = "api key"
response = openai.ChatCompletion.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are an entertainment recommender assistant."},
    {"role": "user", "content": f"I'm feeling {emotion_name}, what music or video should I watch?"}
  ]
)
suggestions = response['choices'][0]['message']['content']
print(suggestions)
```

It can  pair this with Spotify API, YouTube API, or even send it to a web interface via Streamlit/Flask.
---

<p align="center">
  <i>“Your voice tells a story — let AI interpret it and entertain you.”</i>
</p>
