import os
import numpy as np
import librosa
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import pygame
import sounddevice as sd
from scipy.io.wavfile import write

# ========== PARÁMETROS ==========
N_MFCC = 40
MAX_PAD_LEN = 200
AUDIO_GRABADO = "grabacion.wav"
DURACION_GRABACION = 3  # segundos
FS = 44100  # Frecuencia de muestreo

# ========== CARGAR MODELO ==========
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
        messagebox.showwarning("Advertencia", "Selecciona o graba un archivo de audio.")
        return
    try:
        resultado = predecirEmocion(ruta)
        resultado_var.set(f"🎧 Emoción detectada: {resultado}")
    except Exception as e:
        messagebox.showerror("Error en predicción", str(e))

def grabarAudio():
    try:
        resultado_var.set("🎙️ Grabando...")
        root.update()

        # Detener cualquier reproducción de audio en curso
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()

        # Grabar usando stream explícito
        segundos = DURACION_GRABACION
        grabacion = sd.rec(int(segundos * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()

        write(AUDIO_GRABADO, FS, grabacion)

        entrada_audio.set(AUDIO_GRABADO)
        reproducir_btn.config(state=tk.NORMAL)
        predecir_btn.config(state=tk.NORMAL)
        resultado_var.set("✅ Grabación completada. Listo para reproducir o predecir.")

    except Exception as e:
        messagebox.showerror("Error al grabar", str(e))


# ========== INTERFAZ ==========
root = tk.Tk()
root.title("🎤 Detector de Emociones en la Voz")
root.geometry("550x360")
root.configure(bg="#f2f2f2")

entrada_audio = tk.StringVar()
resultado_var = tk.StringVar()

tk.Label(root, text="Archivo de audio seleccionado:", bg="#f2f2f2", font=("Arial", 11)).pack(pady=10)
tk.Entry(root, textvariable=entrada_audio, width=60, font=("Arial", 10), state='readonly').pack(pady=5)

btn_frame = tk.Frame(root, bg="#f2f2f2")
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="🎵 Seleccionar Audio", command=seleccionarAudio, width=20, bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="🎤 Grabar Audio", command=grabarAudio, width=20, bg="#2196F3", fg="white").grid(row=0, column=1, padx=5)

reproducir_btn = tk.Button(root, text="▶ Reproducir", command=reproducirAudio, state=tk.DISABLED, bg="#FF9800", fg="white", width=20)
reproducir_btn.pack(pady=5)

predecir_btn = tk.Button(root, text="🔍 Predecir Emoción", command=mostrarPrediccion, state=tk.DISABLED, bg="#9C27B0", fg="white", width=20)
predecir_btn.pack(pady=5)

tk.Label(root, textvariable=resultado_var, font=("Arial", 14), fg="#333", bg="#f2f2f2").pack(pady=20)

root.mainloop()
