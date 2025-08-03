import os
import numpy as np
import librosa
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import pygame  # para reproducir audio

# ========== PARÁMETROS ==========
N_MFCC = 40
MAX_PAD_LEN = 200

# ========== CARGAR MODELO Y LABEL ENCODER ==========
modelo = load_model("modelo_emociones.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== FUNCIONES ==========
def extraerCaracteristicas(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

def predecirEmocion(audio_path):
    features = extraerCaracteristicas(audio_path)
    features_flat = features.flatten()[np.newaxis, ...]
    prediction = modelo.predict(features_flat)
    clase = le.inverse_transform([np.argmax(prediction)])
    return clase[0]

def seleccionarAudio():
    ruta_audio = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav")])
    if ruta_audio:
        entrada_audio.set(ruta_audio)
        reproducir_btn.config(state=tk.NORMAL)
        predecir_btn.config(state=tk.NORMAL)

def reproducirAudio():
    ruta = entrada_audio.get()
    if os.path.exists(ruta):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(ruta)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error al reproducir", str(e))

def mostrarPrediccion():
    ruta = entrada_audio.get()
    if not ruta:
        messagebox.showwarning("Advertencia", "Selecciona un archivo de audio.")
        return
    try:
        resultado = predecirEmocion(ruta)
        resultado_var.set(f"Emoción detectada: {resultado}")
    except Exception as e:
        messagebox.showerror("Error en predicción", str(e))

# ========== INTERFAZ ==========
root = tk.Tk()
root.title("Detector de Emociones en la Voz")

entrada_audio = tk.StringVar()
resultado_var = tk.StringVar()

tk.Label(root, text="Ruta del archivo de audio:").pack(pady=5)
tk.Entry(root, textvariable=entrada_audio, width=60, state='readonly').pack(pady=5)

tk.Button(root, text="Seleccionar Audio", command=seleccionarAudio).pack(pady=5)
reproducir_btn = tk.Button(root, text="Reproducir Audio", command=reproducirAudio, state=tk.DISABLED)
reproducir_btn.pack(pady=5)

predecir_btn = tk.Button(root, text="Predecir Emoción", command=mostrarPrediccion, state=tk.DISABLED)
predecir_btn.pack(pady=5)

tk.Label(root, textvariable=resultado_var, font=("Arial", 14), fg="blue").pack(pady=10)

root.mainloop()
