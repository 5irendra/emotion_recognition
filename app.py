import streamlit as st
import numpy as np
import librosa
import joblib
import os


st.set_page_config(page_title="Speech Emotion Recognition")

model = joblib.load("final_emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def extract_feature(file_name, mfcc=True, chroma=False, mel=False):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
     
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        
        delta = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta.T, axis=0)

        mfcc_combined = np.hstack((mfccs_mean, delta_mean))
        result = np.hstack((result, mfccs_mean))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
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
