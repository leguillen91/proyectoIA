import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ================================
# CONFIGURACIÓN
# ================================
DATASET_PATH = 'audios'
PRUEBA_PATH = 'pruebas'
CLASES = ['negativas', 'positivas', 'neutro']
N_MFCC = 40
MAX_PAD_LEN = 200

# ================================
# FUNCIONES
# ================================
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

# ================================
# CARGAR DATOS
# ================================
X, y = [], []

for clase in CLASES:
    carpeta = os.path.join(DATASET_PATH, clase)
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.wav'):
            path = os.path.join(carpeta, archivo)
            features = extraerCaracteristicas(path)
            X.append(features)
            y.append(clase)

X = np.array(X)
y = np.array(y)

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Aplanar datos
X_flat = X.reshape(X.shape[0], -1)

# ================================
# DIVIDIR DATOS
# ================================
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_onehot, test_size=0.3, random_state=42)

# ================================
# RED NEURONAL MEJORADA
# ================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ================================
# ENTRENAR
# ================================
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test))
model.save("modelo_emociones.h5")
# ================================
# EVALUACIÓN
# ================================
loss, acc = model.evaluate(X_test, y_test)
print(f"\nPrecisión del modelo: {acc*100:.2f}%")

# ================================
# GRAFICAR PRECISIÓN
# ================================
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión durante el entrenamiento')
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# PREDICCIÓN CON AUDIO NUEVO
# ================================
nuevo_audio = os.path.join(PRUEBA_PATH, 'voztriste.wav')
resultado = predecirEmocion(nuevo_audio, model, le)
print(f"\nPredicción para voz Triste: {resultado}")


