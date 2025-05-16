import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time

# ---------------- Configuration ----------------
DATA_DIR = "ivr_speech_project/data/features"
output_csv = os.path.join(DATA_DIR, "dashboard_data.csv")

mfcc_path = os.path.join(DATA_DIR, "mfcc_features.npy")
labels_path = os.path.join(DATA_DIR, "labels.npy")

# ---------------- Load Data ----------------
X = np.load(mfcc_path, allow_pickle=True)
y = np.load(labels_path, allow_pickle=True)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

max_len = max(x.shape[0] for x in X)
X_padded = np.array([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') for x in X])

# ---------------- Load Models ----------------
hmm_models = joblib.load(os.path.join(DATA_DIR, "hmm_models.pkl"))
dl_model = load_model(os.path.join(DATA_DIR, "speech_dl_model.h5"))

# ---------------- Evaluate Deep Learning Model ----------------
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

y_cat = to_categorical(y_encoded)
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X_padded, y_cat, test_size=0.2, random_state=42)
y_test = np.argmax(y_test_cat, axis=1)

start = time.time()
y_pred_dl = np.argmax(dl_model.predict(X_test), axis=1)
end = time.time()

acc_dl = accuracy_score(y_test, y_pred_dl)
dl_report = classification_report(y_test, y_pred_dl, output_dict=True)
dl_latency = (end - start) / len(X_test)

# ---------------- Evaluate HMM ----------------
def predict_hmm(x):
    scores = {label: model.score(x) for label, model in hmm_models.items()}
    return max(scores, key=scores.get)

start = time.time()
y_pred_hmm = [predict_hmm(x) for x in X_test]
end = time.time()

acc_hmm = accuracy_score(y_test, y_pred_hmm)
hmm_report = classification_report(y_test, y_pred_hmm, output_dict=True)
hmm_latency = (end - start) / len(X_test)

# ---------------- Prepare Data for Power BI ----------------
records = []

for i, label in enumerate(le.classes_):
    records.append({
        "Label": label,
        "Model": "DeepLearning",
        "Precision": dl_report[str(i)]['precision'],
        "Recall": dl_report[str(i)]['recall'],
        "F1-Score": dl_report[str(i)]['f1-score'],
        "Accuracy": acc_dl,
        "Latency": dl_latency
    })
    records.append({
        "Label": label,
        "Model": "HMM",
        "Precision": hmm_report[str(i)]['precision'],
        "Recall": hmm_report[str(i)]['recall'],
        "F1-Score": hmm_report[str(i)]['f1-score'],
        "Accuracy": acc_hmm,
        "Latency": hmm_latency
    })

df = pd.DataFrame.from_records(records)
df.to_csv(output_csv, index=False)
print(f"✅ Dashboard data exported to {output_csv}")
