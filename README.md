
Estas librerías permiten:

* Cargar y procesar archivos de audio (`librosa`)
* Trabajar con arrays (`numpy`)
* Codificar etiquetas y dividir el dataset (`sklearn`)
* Crear y entrenar redes neuronales (`tensorflow`)
* Guardar modelos y etiquetas (`pickle`)
* Graficar (`matplotlib`)

---

##  **Configuración**

```python
DATASET_PATH = 'audios'
PRUEBA_PATH = 'pruebas'
CLASES = ['negativas', 'positivas', 'neutro']
N_MFCC = 40
MAX_PAD_LEN = 200
```

* `DATASET_PATH`: carpeta con subcarpetas de audios clasificados en emociones.
* `CLASES`: clases de emociones.
* `N_MFCC`: número de coeficientes MFCC (características del audio).
* `MAX_PAD_LEN`: longitud fija para uniformar los vectores de características.

---

## **Funciones clave**

### 🔹 `extraerCaracteristicas`

```python
def extraerCaracteristicas(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    ...
    return mfcc
```

* Extrae MFCC (características del audio que representan el timbre).
* Asegura que todos los vectores tengan el mismo tamaño (relleno o corte).

---

### `predecirEmocion`

```python
def predecirEmocion(audio_path, model, le):
    features = extraerCaracteristicas(audio_path)
    ...
    return clase[0]
```

* Extrae características del nuevo audio.
* Usa el modelo entrenado para predecir su clase emocional.
* `le.inverse_transform` decodifica la predicción al nombre de la clase.

---

##  **Carga y preparación de datos**

```python
X, y = [], []
for clase in CLASES:
    carpeta = os.path.join(DATASET_PATH, clase)
    ...
```

* Carga todos los archivos `.wav` en cada subcarpeta (emociones).
* Extrae características de cada uno y los etiqueta.

### Codificación:

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)
```

* Codifica las emociones (`positiva`, `negativa`, `neutro`) como enteros y luego a *one-hot vectors*.

### Aplanar los MFCC:

```python
X_flat = X.reshape(X.shape[0], -1)
```

* Convierte cada matriz MFCC 2D en un vector 1D para alimentar la red.

---

## **División de datos**

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

* Separa datos en entrenamiento (70%) y prueba (30%).

---

## **Red Neuronal**

```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
```

* Red neuronal con 3 capas ocultas.
* `Dropout`: evita sobreajuste.
* `Softmax`: última capa para clasificación multiclase.

---

##  **Entrenamiento**

```python
history = model.fit(...)
```

* Se entrena el modelo durante 50 épocas, con validación en datos de prueba.
* Se guarda el modelo (`.h5`) y el codificador de etiquetas (`.pkl`).

---

##  **Evaluación**

```python
loss, acc = model.evaluate(X_test, y_test)
print(f"\nPrecisión del modelo: {acc*100:.2f}%")
```

* Imprime la precisión sobre los datos de prueba.

---

##  **Graficar entrenamiento**

```python
plt.plot(history.history['accuracy'], ...)
```

* Muestra cómo evolucionó la precisión durante las épocas.

---

##  **Predicción de audio nuevo**

```python
nuevo_audio = os.path.join(PRUEBA_PATH, 'vozNeutral.wav')
resultado = predecirEmocion(nuevo_audio, model, le)
print(f"\nPredicción para voz Neutral: {resultado}")
```

* Usa el modelo para predecir la emoción de un archivo nuevo.

---
