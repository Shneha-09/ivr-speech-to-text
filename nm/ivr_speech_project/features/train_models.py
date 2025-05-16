import numpy as np
import os
import joblib
from hmmlearn import hmm
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- Configuration ----------------
DATA_DIR = "ivr_speech_project/data/features"
mfcc_path = os.path.join(DATA_DIR, "mfcc_features.npy")
labels_path = os.path.join(DATA_DIR, "labels.npy")

if not os.path.exists(mfcc_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("❌ Feature or label files not found. Check paths!")

X = np.load(mfcc_path, allow_pickle=True)
y = np.load(labels_path, allow_pickle=True)

# ---------------- Preprocessing ----------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

max_len = max(x.shape[0] for x in X)
X_padded = np.array([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') for x in X])

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_cat, test_size=0.2, random_state=42)

# ---------------- HMM Training ----------------
def train_hmms(X, y, n_states=5):
    models = {}
    for label in np.unique(y):
        X_class = [x for x, l in zip(X, y) if l == label]
        X_concat = np.concatenate(X_class)
        lengths = [len(x) for x in X_class]
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
        model.fit(X_concat, lengths)
        models[label] = model
    return models

hmm_models = train_hmms(X, y_encoded)
joblib.dump(hmm_models, os.path.join(DATA_DIR, "hmm_models.pkl"))
print("✅ HMM training complete.")

# ---------------- Deep Learning Model ----------------
def build_dl_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_dl_model(X_train.shape[1:], y_cat.shape[1])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
model.save(os.path.join(DATA_DIR, "speech_dl_model.h5"))
print("✅ Deep Learning training complete.")
