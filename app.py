import streamlit as st
import numpy as np
import librosa
import joblib
import io

# Page configuration
st.set_page_config(page_title="Speech Emotion Recognition")

# Load model and label encoder
model = joblib.load("final_emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature extraction function
def extract_feature(uploaded_file, mfcc=True, chroma=True, mel=True):
    result = np.array([])

    # Load audio file from memory
    audio_data, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')

    if chroma:
        stft = np.abs(librosa.stft(audio_data))

    if mfcc:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        result = np.hstack((result, mfccs_mean))

    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

    return result

# Streamlit UI
st.title("Speech Emotion Recognition")
st.markdown("Upload a WAV audio file to detect the emotion.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    try:
        features = extract_feature(uploaded_file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        prediction = model.predict(features)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Emotion: {predicted_emotion}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
