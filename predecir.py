import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# ========== PARÁMETROS ==========
N_MFCC = 40
MAX_PAD_LEN = 200
PRUEBA_PATH = 'pruebas'


def extraerCaracteristicas(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

def predecirEmocion(audio_path, model, le):
    features = extraerCaracteristicas(audio_path)
    features_flat = features.flatten()[np.newaxis, ...]
    prediction = model.predict(features_flat)
    clase = le.inverse_transform([np.argmax(prediction)])
    return clase[0]

# ========== CARGAR MODELO Y LABEL ENCODER ==========
modelo = load_model("modelo_emociones.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== PREDICCIÓN ==========
nuevo_audio = os.path.join(PRUEBA_PATH, "vozalegre2.wav")
resultado = predecirEmocion(nuevo_audio, modelo, le)

print(f"Predicción para voz: {resultado}")

