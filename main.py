'''
Author: Aditti Ramsisaria
Description: Predict emotion from speech audio trained on RAVDESS
Filename identifiers for .wav files:
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 
             06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). 
    NOTE: There is no strong intensity for the ‘neutral’ emotion.
    Statement (01 = “Kids are talking by the door”, 
               02 = “Dogs are sitting by the door”).
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, 
                     even numbered actors are female).
'''

import time
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# path to current folder
path_c = "C:\\Users\\aditi\\Desktop\\6th semester\\16264 - Humanoids\\SER\\"
# path to dataset
path_d = path_c + "speech-emotion-recognition-ravdess-data\\"

# define emotions dictionary
emotions = {
    "01" : "neutral",
    "02" : "calm",
    "03" : "happy",
    "04" : "sad",
    "05" : "angry",
    "06" : "fearful",
    "07" : "disgust",
    "08" : "surprise"
}

# subset of emotions used
observed_emotions = ["calm", "happy", "fearful", "disgust"]

print("Emotions subset: ", observed_emotions)

# extract features from audio - mfcc, mel, chroma
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if chroma:
            # initialize
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, 
                                                 n_mfcc=40).T, axis=0)
            # append to result
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, 
                                                sr=sample_rate).T,axis=0)
            # append to result
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, 
                                                sr=sample_rate).T,axis=0)
            # append to result
            result = np.hstack((result, mel))
    return result

# load data and extract features, assign labels
# return train and test sets
def load_data(test_size=0.2):
    x = []  # features
    y = []  # labels
    # for each .wav file 
    start_time = time.time()
    for file in glob.glob(path_d + "Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        # get emotion from filename as specified in identifiers
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # append to list of features and emotion labels
        x.append(feature)
        y.append(emotion)
    # change x to np array
    x = np.array(x)
    print("\nTime taken for feature extraction : ", time.time() - start_time)
    return train_test_split(x, y, test_size=test_size, random_state=9)

# get training and test data
x_train, x_test, y_train, y_test = load_data(test_size=0.2)

# preprocessing: scale data - MLP sensitive to scaling in features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("Number of training samples: ", x_train.shape)
print("Number of testing samples: ", x_test.shape)

# generate MLP classifier model
# if using adam, learning_rate='adaptive', epsilon=1e-08,
# learning_rate_init=0.0001, max_iter=5000
optimizer = "lbfgs"
model = MLPClassifier(solver=optimizer, alpha=0.1, 
                      hidden_layer_sizes=(300,), 
                      random_state=9)

# train the model
start_time = time.time()
model.fit(x_train, y_train)
print("\nOptimizer used: ", optimizer)
print("Time taken for training: ", time.time() - start_time)
print("Loss computed: ", model.loss_)

# predict for test set
y_pred = model.predict(x_test)

# calculate accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Testing Accuracy: {:.2f}%".format(accuracy * 100))

        