import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.sancho_model import GPTLanguageModel, device, vocab_size, n_embd

# 1. Cargar el modelo
model = GPTLanguageModel().to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ckpt.pt')
if torch.cuda.is_available():
    model.load_state_dict(torch.load(ckpt_path))
else:
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

print("Modelo cargado. Generando visualizaciones...")

# 2. Extraer los Pesos
# Positional Embeddings: Qué aprendió sobre la posición de las letras
pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()

# Token Embeddings: Qué aprendió sobre cada letra
tok_emb = model.token_embedding_table.weight.detach().cpu().numpy()

# 3. Dibujar
plt.figure(figsize=(15, 6))

# Gráfico A: Embeddings Posicionales (El "Ritmo")
plt.subplot(1, 2, 1)
# Visualizamos solo las primeras 100 posiciones y 100 dimensiones para ver detalle
sns.heatmap(pos_emb[:100, :100], cmap="viridis")
plt.title(f"Positional Embeddings (Primeros 100x100)\nEl modelo aprende patrones diagonales/ondulados")
plt.xlabel("Dimensión del Embedding")
plt.ylabel("Posición en la frase")

# Gráfico B: Embeddings de Tokens (El "Significado")
plt.subplot(1, 2, 2)
# Visualizamos todos los caracteres (aprox 70)
sns.heatmap(tok_emb, cmap="magma")
plt.title(f"Token Embeddings ({vocab_size} caracteres)\nCada fila es la 'personalidad' de una letra")
plt.xlabel("Dimensión del Embedding")
plt.ylabel("Índice del Carácter")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'media', 'model_internals.png')
plt.savefig(output_path)
print("¡Imagen guardada como 'model_internals.png'!")
plt.show()