# Simple RNN for IMDb Sentiment Analysis

## 📌 Introduction

This project is an end-to-end deep learning implementation using a **Simple Recurrent Neural Network (Simple RNN)** for sentiment analysis on the IMDb movie reviews dataset. The objective is to classify movie reviews as **positive** or **negative** based on their textual content.

The primary goal of this project is **not to achieve state-of-the-art accuracy**, but to **implement, analyze, and understand the behavior and limitations of a Simple RNN** when applied to real-world text data.

The IMDb dataset contains approximately **50,000 labeled reviews**, making it a substantial and commonly used benchmark for sentiment classification tasks.

---

## 🏗️ Project Architecture

![Project Architecture](images/project_architecture.png)

**Workflow overview:**

1. Load and preprocess IMDb review data
2. Transform text into numerical representations
3. Train a Simple RNN model with an embedding layer
4. Save the trained model in `.keras` format
5. Build a Streamlit web application for user interaction

---

## 🧠 Simple RNN Model Components

The model consists of two core layers:

### 1️⃣ Embedding Layer

* Converts word indices into dense vector representations
* Enables the model to work with numerical inputs instead of raw text
* Learns word embeddings during training

### 2️⃣ Simple RNN Layer

* Processes sequences
* Attempts to capture temporal dependencies in text
* Limited in handling long sequences due to vanishing gradients

---

## 📂 Loading the IMDb Dataset

The IMDb dataset is directly available via TensorFlow.

```python
from tensorflow.keras.datasets import imdb

max_features = 10000  # vocabulary size

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
```

* Training samples: 25,000
* Testing samples: 25,000
* Labels:

  * `1` → Positive
  * `0` → Negative

---

## 🔍 Exploring and Decoding Reviews

Each review is represented as a sequence of integers, where each integer maps to a word index in the vocabulary.

### Decoding Integer-Encoded Reviews

```python
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
print(decoded_review)
```

> ℹ️ The reason for subtracting `3` during decoding is explained in detail
> [here](https://theshoaibakthar.github.io/posts/2025/12/why-subtract-3-in-word-index-in-IMDb-dataset-tensorflow/).

---

## 📏 Padding Sequences

Neural networks require inputs of uniform length. Since reviews vary in size, padding is applied:

```python
from tensorflow.keras.preprocessing import sequence

max_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
```

* Pre-padding with zeros
* Maximum sequence length: **500 words**

---

## 🏋️ Model Training

### Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## ⏱️ Early Stopping

To prevent overfitting and reduce unnecessary training, EarlyStopping is used:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

---

### Training the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

* 80% training / 20% validation split
* Training may stop early if validation loss stops improving

---

## 💾 Model Saving

The trained model is saved in `.keras` format:

```python
model.save("SIMPLE_RNN_IMDb_Model.keras")
```

This allows easy reuse for prediction and deployment.

---

## 🔮 Model Prediction

A separate notebook (`Prediction.ipynb`) handles prediction logic.

### Prediction Steps

1. Load the trained model
2. Preprocess user input
3. Generate prediction score
4. Classify sentiment

### Classification Logic

* Score > 0.5 → **Positive**
* Score ≤ 0.5 → **Negative**

---

### Example Prediction

**Input review:**

> *The movie was fantastic, the acting was great, and the plot was thrilling.*

**Output:**

* Sentiment: **Positive**
* Prediction score: **0.91**

---

## ⚠️ Model Limitations & Observations

Despite tuning and configuration changes, the model **fails to generalize reliably** on unseen examples.

### Modifications Made

* Removed activation from the embedding layer
* Increased EarlyStopping patience from 5 → 8
* Increased epochs from 10 → 20

### Incorrect Prediction Example

**Input review:**

> *This movie was bad. I don't like it.*

**Output:**

* Sentiment: **Positive**
* Prediction score: **0.87**

---

### Key Takeaway

This behavior highlights the **limitations of Simple RNNs**:

* Struggle with long-term dependencies
* Suffer from vanishing gradient issues
* Fail to capture deeper semantic meaning
* Not suitable for long or complex text reviews

While architectures like **LSTM or GRU** would significantly improve performance, this project intentionally focuses on **Simple RNN implementation and analysis**.

---

## 🌐 Streamlit Web Application

A Streamlit-based web app is built for user interaction.

### Features

* Text input for movie reviews
* Sentiment classification on button click
* Displays sentiment label and prediction score

![Streamlit App](images/streamlit-app.png)

---

## 🚀 Next Steps

* Explore LSTM / GRU architectures
* Improve generalization
* Compare performance across RNN variants

---

## 📌 Conclusion

This project successfully demonstrates the **end-to-end implementation of a Simple RNN** for sentiment analysis while clearly exposing its limitations. It serves as a strong foundational exercise for understanding why more advanced sequence models are required in modern NLP tasks.
