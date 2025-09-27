# microphone-classifier.py
import tensorflow as tf
import librosa
import numpy as np
import joblib
import sounddevice as sd
import wavio
import time

# Load the model
model = tf.keras.models.load_model("trained_model.h5")
labelencoder = joblib.load("label_encoder.joblib")

# Microphone parameters
duration = 3  # Recording duration (seconds)
fs = 44100    # Sampling rate (Hz)

# Function 1: Records audio and writes it to a file
def record_and_save_audio(filename, duration=3, fs=44100):
    print(f"Recording {filename}...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is complete
    wavio.write(filename, recording, fs, sampwidth=2)  # Save the audio file
    print(f"{filename} has been recorded.")

# Function 2: Classifies the recorded audio file
def classify_audio(filename):
    # Classify the new audio file
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfcc_features.T, axis=0)

    # Reshape the input to the expected shape of the model (1, 40)
    mfccs_scaled_features = np.expand_dims(mfccs_scaled_features, axis=0)

    # Make predictions
    result_array = model.predict(mfccs_scaled_features)

    result_classes = ["ambulance", "firetruck", "traffic sound"]

    # Print prediction probabilities for each class
    print("Classes and Prediction Probabilities:")
    for i, result_class in enumerate(result_classes):
        print(f"{result_class}: {result_array[0][i]:.2f}")

    # Check the highest prediction probability
    max_confidence = np.max(result_array[0])  # Highest prediction probability

    # If max prediction probability is below 80%, return "traffic sound"
    if max_confidence < 0.80:
        print("\nResult: traffic sound (Prediction Probability is below 80%)")
    else:
        # If above 80%, perform normal classification
        result_index = np.argmax(result_array[0])
        confidence = result_array[0][result_index]  # Highest prediction probability

        # Print the result and the highest prediction probability
        print(f"\nResult: {result_classes[result_index]} (Highest Prediction Probability: {confidence:.2f})")

# Loop to record audio every 3 seconds and classify it
try:
    i = 1
    while True:
        filename = f"sounds/recording_{i}.wav"  # Name of the audio file to be saved
        record_and_save_audio(filename, duration, fs)  # Get audio recording
        classify_audio(filename)  # Classify the recorded audio
        time.sleep(3)  # Repeat the loop every 3 seconds
        i += 1
except KeyboardInterrupt:
    print("Audio recording and classification has been stopped.")
