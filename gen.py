import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Importamos todo lo necesario de sancho.py
# Nota: Esto ejecutará el código de nivel superior de sancho.py (carga de datos, etc.)
# pero NO el entrenamiento gracias al if __name__ == '__main__':
from sancho_model import GPTLanguageModel, encode, decode, device, block_size, n_embd, n_head, n_layer, vocab_size

print("\nsancho-mini generando texto :D")

# Configuración
temperature = 0.4 # Un poco más alto para más creatividad
top_k = 200

# Cargar el modelo
m = GPTLanguageModel()
m = m.to(device)

if os.path.exists('ckpt.pt'):
    print("Cargando modelo entrenado...")
    m.load_state_dict(torch.load('ckpt.pt', map_location=device))
else:
    print("ADVERTENCIA: No se encontró 'ckpt.pt'. Usando modelo sin entrenar.")

m.eval()

# Contexto inicial: un salto de línea o un espacio es seguro
start_str = "\n" 
context = torch.tensor([encode(start_str)], dtype=torch.long, device=device)

print(start_str, end='')
sys.stdout.flush()

with torch.no_grad():
    for _ in range(500): # Generar 500 caracteres
        # Recortar el contexto si es necesario
        idx_cond = context[:, -block_size:]
        
        # Obtener predicciones
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :] # Último token
        
        # Ajustar temperatura
        logits = logits / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Probabilidades
        probs = F.softmax(logits, dim=-1)
        
        # Muestrear siguiente token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Decodificar e imprimir
        char = decode(idx_next[0].tolist())
        print(char, end='')
        sys.stdout.flush()
        
        # Actualizar contexto
        context = torch.cat((context, idx_next), dim=1)

print("\n")

