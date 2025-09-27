# train-model.py
import tensorflow as tf
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
import joblib  # To save the model
import h5py  # To save the model file

# Path to the directories where audio files are located
audio_dataset_path = "sounds/"  # Main folder
class_folders = ['ambulance', 'firetruck', 'traffic']  # Subfolders

# Feature extraction function
def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# List to collect features and labels
extracted_features = []

# Process audio files by iterating through each folder
for class_label in class_folders:
    folder_path = os.path.join(audio_dataset_path, class_label)  # Subfolder path
    for file_name in tqdm(os.listdir(folder_path)):  # For each file in the folder
        if file_name.endswith('.wav'):  # Only process .wav files
            file_path = os.path.join(folder_path, file_name)  # Full file path
            data = feature_extractor(file_path)  # Extract features
            extracted_features.append([data, class_label])  # Add features and label

# After processing the data, you can convert it to a numpy array or pandas DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=["features", "class"])

x = np.array(extracted_features_df["features"].tolist())
y = np.array(extracted_features_df["class"].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# Separate the training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Model layers
num_labels = 3
model = Sequential()
model.add(Dense(125, input_shape=(40,)))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(250))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(125))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

# Train the model
epochscount = 30
num_batch_size = 32
model.fit(xtrain, ytrain, batch_size=num_batch_size, epochs=epochscount, validation_data=(xtest, ytest), verbose=1)

# Save the model
model.save("trained_model.h5")
joblib.dump(labelencoder, "label_encoder.joblib")
