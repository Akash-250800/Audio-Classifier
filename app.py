# app.py
import streamlit as st
from utils import audio_to_mel, load_model
import torch

st.title(" Music Genre Classifier")
st.write("Upload a `.wav` file and I'll tell you the genre!")

# Genre labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Extracting features and predicting..."):
        mel_tensor = audio_to_mel(uploaded_file)
        model = load_model("model/spectrogram_cnn.pth", num_classes=len(genres))
        prediction = model(mel_tensor)
        predicted_index = torch.argmax(prediction, dim=1).item()
        predicted_genre = genres[predicted_index]

    st.success(f" Predicted Genre: **{predicted_genre.upper()}**")
