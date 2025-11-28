# SanchoGPT 🛡️📖

**SanchoGPT** es un modelo de lenguaje compacto basado en la arquitectura GPT (Generative Pre-trained Transformer), entrenado específicamente con el texto de *"El Ingenioso Hidalgo Don Quijote de la Mancha"* de Miguel de Cervantes.

El objetivo de este proyecto es explorar cómo un modelo pequeño puede aprender el estilo y vocabulario del español antiguo a nivel de caracteres.

![Demo](media/Animation.gif)

## 🚀 Características

- **Arquitectura GPT**: Implementación desde cero en PyTorch (Self-Attention, Feed-Forward, LayerNorm).
- **Entrenamiento a nivel de carácter**: El modelo genera texto letra por letra.
- **Visualización**: Herramientas para inspeccionar los embeddings y la arquitectura.
- **Exportación**: Soporte para exportar a ONNX y visualizar en 3D.

## 🛠️ Instalación

Asegúrate de tener Python instalado. Las dependencias principales son:

```bash
pip install torch matplotlib seaborn numpy
```

## 💻 Uso

El proyecto consta de varios scripts organizados en carpetas:

### 1. Entrenamiento (`model/sancho_model.py`)
Entrena el modelo desde cero.

```bash
python model/sancho_model.py
```

### 2. Generación de Texto (`gen.py`)
Carga el modelo entrenado (`model/ckpt.pt`) y genera texto al estilo de Cervantes.

```bash
python gen.py
```

### 3. Visualización de Embeddings (`visualization/view.py`)
Genera mapas de calor para visualizar qué ha aprendido el modelo.
- Genera: `media/model_internals.png`

```bash
python visualization/view.py
```

![Embeddings](media/model_internals.png)

### 4. Exportación de Vectores 3D (`visualization/3dview.py`)
Exporta los embeddings a archivos TSV (`visualization/vectors.tsv` y `visualization/metadata.tsv`) para visualizarlos en [TensorFlow Projector](https://projector.tensorflow.org/).

```bash
python visualization/3dview.py
```

### 5. Exportación a ONNX (`visualization/onnx.py`)
Exporta la arquitectura del modelo al formato ONNX para visualizar en [Netron](https://netron.app/).
- Genera: `visualization/sancho_architecture.onnx`

```bash
python visualization/onnx.py
```

![Arquitectura](media/sancho_architecture.onnx.png)

## 📂 Estructura del Proyecto

- **`model/`**:
    - `sancho_model.py`: Definición del modelo y entrenamiento.
    - `datos_sancho_mini.txt`: Dataset.
    - `ckpt.pt`: Checkpoint del modelo.
- **`visualization/`**:
    - `view.py`: Visualización 2D.
    - `3dview.py`: Exportación 3D.
    - `onnx.py`: Exportación ONNX.
- **`media/`**: Imágenes y GIFs del proyecto.
- `gen.py`: Script principal de generación.
