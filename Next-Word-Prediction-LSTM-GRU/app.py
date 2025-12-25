import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load tokenizer
# -----------------------------
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# -----------------------------
# Cache models for performance
# -----------------------------
@st.cache_resource
def load_selected_model(model_name):
    if model_name == "LSTM":
        return load_model("next_word_lstm.keras")
    elif model_name == "GRU":
        return load_model("next_word_gru.keras")

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Keep last max_sequence_len - 1 tokens
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Reverse lookup
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Unknown"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Next Word Prediction (LSTM / GRU)")

model_choice = st.selectbox(
    "Select Model",
    ["LSTM", "GRU"]
)

input_text = st.text_input(
    "Enter a sequence of words",
    "To be or not to"
)

if st.button("Predict Next Word"):
    model = load_selected_model(model_choice)

    # Input shape: (None, sequence_length)
    max_sequence_len = model.input_shape[1] + 1

    next_word = predict_next_word(
        model,
        tokenizer,
        input_text,
        max_sequence_len
    )

    st.success(f"**Predicted next word ({model_choice}):** {next_word}")
