import librosa
import numpy as np
import joblib
import os

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

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


model = joblib.load('final_emotion_model.pkl')


file_path = 'test.wav'  


features = extract_feature(file_path, mfcc=True, chroma=True, mel=True).reshape(1, -1)
predicted_emotion = model.predict(features)

print("Predicted Emotion:", predicted_emotion[0])
