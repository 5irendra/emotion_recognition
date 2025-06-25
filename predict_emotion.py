import os
import argparse
import librosa
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load('final_emotion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature extractor
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    result = np.array([])

    if chroma:
        stft = np.abs(librosa.stft(X))

    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        result = np.hstack((result, np.mean(mfccs.T, axis=0)))

    if chroma:
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        result = np.hstack((result, np.mean(chroma.T, axis=0)))

    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        result = np.hstack((result, np.mean(mel.T, axis=0)))

    return result

# Predict emotion from one file
def predict_emotion(file_path):
    features = extract_feature(file_path).reshape(1, -1)
    prediction = model.predict(features)
    emotion = label_encoder.inverse_transform(prediction)[0]
    return emotion

# Main CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion prediction from a .wav file or a folder of .wav files")
    parser.add_argument("--file", type=str, help="Path to a single .wav file")
    parser.add_argument("--folder", type=str, help="Path to folder containing multiple .wav files")

    args = parser.parse_args()

    if args.file:
        if args.file.endswith(".wav") and os.path.isfile(args.file):
            print(f"Predicting emotion for: {args.file}")
            try:
                emotion = predict_emotion(args.file)
                print(f"Predicted Emotion: {emotion}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid file path. Please provide a valid .wav file.")

    elif args.folder:
        if os.path.isdir(args.folder):
            print(f"Predicting emotions in folder: {args.folder}\n")
            for filename in os.listdir(args.folder):
                if filename.endswith(".wav"):
                    file_path = os.path.join(args.folder, filename)
                    try:
                        emotion = predict_emotion(file_path)
                        print(f"{filename} → {emotion}")
                    except Exception as e:
                        print(f"{filename} → Error: {e}")
        else:
            print("Folder does not exist or path is incorrect.")
    else:
        print("No input provided. Please use --file or --folder to specify the test data.")

