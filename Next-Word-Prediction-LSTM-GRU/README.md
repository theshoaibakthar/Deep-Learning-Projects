# Next Word Prediction using LSTM & GRU

## Introduction

This project focuses on building a **deep learning–based next word prediction system** using sequence models. The application predicts the most likely next word given a sequence of words, using **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** networks.
The complete solution covers data preprocessing, model training, evaluation, and deployment using **Streamlit Cloud**.

---

## Project Workflow

The project follows these major steps:

* **Data Collection** – Using Shakespeare’s *Hamlet* text from the NLTK Gutenberg corpus
* **Data Preprocessing** – Tokenization, sequence generation, padding, and train-test split
* **Model Building** – Separate LSTM and GRU architectures
* **Model Training** – Training with validation and early stopping (where applicable)
* **Model Evaluation** – Predicting next words for unseen sequences
* **Deployment** – Interactive Streamlit web app hosted on Streamlit Cloud

---

## Data Collection

The dataset is obtained from the **NLTK Gutenberg corpus**, specifically *Shakespeare’s Hamlet*.
The raw text is extracted and saved locally for preprocessing and training.

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w') as file:
    file.write(data)
```

---

## Data Preprocessing

The text is converted to lowercase and tokenized using Keras `Tokenizer`. Each word is mapped to an integer index, and the vocabulary size is calculated.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
```

---

## Creating Input Sequences

The text is split line by line, and **n-gram sequences** are created. This allows the model to learn how words evolve sequentially.

```python
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])
```

---

## Padding and Train-Test Split

All sequences are padded to the same length so they can be processed by the RNN models.
The dataset is split into training and testing sets using an 80-20 ratio.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

max_sequence_len = max(len(x) for x in input_sequences)
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Model Building and Training

### LSTM Model

The LSTM model consists of:

* Embedding layer
* Two stacked LSTM layers
* Dropout for regularization
* Dense softmax output layer

```python
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
```

The model is trained using **categorical cross-entropy loss** and the **Adam optimizer**.
Early stopping is explored to prevent overfitting, though further tuning is required for this dataset.

> The same training process is followed for the GRU model, replacing LSTM layers with GRU layers.

---

## Model Evaluation

A custom function is created to predict the next word given an input sequence.
The function preprocesses the input text, applies padding, and selects the word with the highest predicted probability.

```python
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_index = model.predict(token_list).argmax(axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
```

---

## Saving the Model

After training, the models and tokenizer are saved for reuse during deployment.

```python
model.save('next_word_lstm.keras')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)
```

---

## Streamlit Application

The Streamlit app provides an interactive interface where users can:

* Select **LSTM or GRU** model
* Enter a sequence of words
* Get real-time next word predictions

To improve performance, models are loaded dynamically using Streamlit’s caching mechanism.

---

## Deployment

The application is deployed using **Streamlit Cloud** by connecting the GitHub repository.
All dependencies are specified in a `requirements.txt` file, allowing Streamlit Cloud to automatically build and deploy the app.

🔗 **Live App:** [Link here]

![streamlit app](https://theshoaibakthar.github.io/images/portfolio/Next-word-Prediction/Pasted%20image%2020251225184109.png)

---

## Conclusion

This project demonstrates my ability to build and deploy an end-to-end NLP solution, from raw text processing to a live web application. By implementing both LSTM and GRU models and deploying them through Streamlit Cloud, the project highlights practical experience in deep learning, sequence modeling, and machine learning deployment.
