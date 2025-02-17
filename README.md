# Fine-Tuned BERT for IMDB Sentiment Analysis 🎬🤖

Welcome to the **Fine-Tuned BERT for IMDB Sentiment Analysis** project! This repository demonstrates how to fine-tune the BERT model to classify movie reviews from the IMDB dataset as positive or negative. Whether you're a movie enthusiast, data scientist, or AI researcher, this project offers valuable insights into leveraging transformer models for sentiment analysis. 🎥📊

## Table of Contents 📑

- [Overview](#overview) 🌟
- [Model Architecture](#model-architecture) 🏗️
- [Training Procedure](#training-procedure) 🏋️‍♂️
- [Evaluation and Performance](#evaluation-and-performance) 📈
- [How to Use](#how-to-use) 🛠️
- [Author](#author) 🧑‍💻
- [License](#license) 📜

## Overview 🌟

This project focuses on fine-tuning the `bert-base-uncased` model on the IMDB movie reviews dataset, which consists of 25,000 training samples and 25,000 testing samples, evenly split between positive and negative sentiments. The goal is to accurately classify the sentiment of movie reviews. 🎬📈

## Model Architecture 🏗️

- **Backbone**: [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805) 🧠
- **Classification Head**: A single linear layer on top of the pooled `[CLS]` token output for binary classification. 🔗

*Why BERT?* BERT's bidirectional training allows it to understand the context from both directions in a sentence, making it adept at capturing the nuances in movie reviews. 🎥🧠

## Training Procedure 🏋️‍♂️

1. **Data Loading**: 📥
   - The IMDB dataset was loaded with an even split of positive and negative reviews. 🎬
2. **Preprocessing**: 🧹
   - Tokenization using the BERT tokenizer (`bert-base-uncased`), truncating/padding to a fixed length (e.g., 128 tokens). 📝
3. **Hyperparameters**: ⚙️
   - Learning Rate: 5e-5 🔧
   - Batch Size: 8 🧳
   - Epochs: 3 🔄
   - Optimizer: Adam 🏃‍♂️
   - Loss Function: Sparse Categorical Cross-entropy ⚖️
4. **Hardware**: 💻
   - Fine-tuned on a GPU (e.g., Google Colab or a local machine with CUDA). 🖥️
5. **Validation**: ✅
   - Periodic evaluation on the validation set to monitor accuracy and loss. 📊

The entire fine-tuning process is documented in the [notebook](https://github.com/harshhmaniya/Fine-Tuned-BERT-Sentiment-Analysis/blob/main/imdb_reviews_bert.ipynb) included in this repository. 📓

## Evaluation and Performance 📈

- **Accuracy**: ~93% on the IMDB test set. 🎯

This performance indicates that the model effectively handles most typical movie reviews. However, it might still face challenges with highly sarcastic or context-dependent reviews. 🎬🤔

## How to Use 🛠️

To utilize this model for sentiment analysis in Python using TensorFlow: 🐍

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load tokenizer and model
model_name = "harshhmaniya/fine-tuned-bert-imdb-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Example text
review_text = "I absolutely loved this movie! The plot was gripping and the acting was top-notch."

# Prepare input
inputs = tokenizer(review_text, return_tensors="tf", truncation=True, padding=True)

# Perform inference
outputs = model(inputs)
logits = outputs.logits

# Convert logits to probabilities (softmax)
probs = tf.nn.softmax(logits, axis=-1)
pred_class = tf.argmax(probs, axis=-1).numpy()[0]

# Interpret results
label_map = {0: "Negative", 1: "Positive"}
print(f"Review Sentiment: {label_map[pred_class]}")
```

This script will output the sentiment of the provided `review_text` as either "Positive" or "Negative". 📝👍👎

## Author 🧑‍💻

- **Name**: Harsh Maniya 🧑‍💻
- **Hugging Face Model Repository**: [fine-tuned-bert-imdb-sentiment-analysis](https://huggingface.co/harshhmaniya/fine-tuned-bert-imdb-sentiment-analysis) 🌐

## License 📜

This project is licensed under the MIT License. See the [LICENSE](https://github.com/harshhmaniya/Fine-Tuned-BERT-Sentiment-Analysis/blob/main/LICENSE) file for details. 📄
