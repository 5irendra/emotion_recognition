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
label_encoder = joblib.load('label_encoder.pkl')

folder_path = 'test_folder' 

print("Predicting emotions for all .wav files in folder:", folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        try:
            features = extract_feature(file_path, mfcc=True, chroma=True, mel=True).reshape(1, -1)
            prediction = model.predict(features)

            emotion = label_encoder.inverse_transform(prediction)[0]

            print(f"{filename} → Predicted Emotion: {emotion}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

