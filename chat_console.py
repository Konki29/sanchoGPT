import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Importamos la arquitectura (igual que antes)
from model.sancho_model import GPTLanguageModel, encode, decode, device, block_size, n_embd, n_head, n_layer, vocab_size, stoi

# Si el carácter 'c' no existe, usa el espacio ' ' (o el token que corresponda al espacio)
encode = lambda s: [stoi.get(c, stoi.get(' ', 0)) for c in s]
print("\n--- SANCHO CHAT v1.0 ---\n")

# Configuración
temperature =0.5 # Un poco más baja para que sea más coherente respondiendo
top_k = 200

# 1. CARGAR EL MODELO NUEVO (El del chat)
# Asegúrate de que aquí apuntas al archivo que vas a generar con el entrenamiento
ckpt_path = 'sancho_chat.pt' 

m = GPTLanguageModel()
m = m.to(device)

if os.path.exists(ckpt_path):
    print(f"Cargando cerebro de chat ({ckpt_path})...")
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print(f"¡ERROR! No encuentro '{ckpt_path}'. ¿Ya ejecutaste el entrenamiento?")
    sys.exit()

m.eval()

# Definimos los tokens especiales (TIENEN QUE SER LOS MISMOS QUE USES AL ENTRENAR)
TOKEN_USER = "\n<|usuario|>\n"
TOKEN_BOT = "\n<|sancho|>\n"
TOKEN_END = "<|fin|>" # O el token que decidas usar para cortar

print("Escribe 'salir' para terminar.")

# Bucle infinito de conversación
while True:
    # 1. Leer lo que tú dices
    user_input = input("\nTú: ")
    if user_input.lower() == 'salir':
        break

    # 2. Formatear el prompt (Esto es lo que ve el modelo)
    # Le damos la estructura exacta que aprendió en el JSON
    prompt_str = f"{TOKEN_USER}{user_input}{TOKEN_BOT}"
    
    # Codificar a números
    context = torch.tensor([encode(prompt_str)], dtype=torch.long, device=device)

    print("Sancho: ", end='')
    sys.stdout.flush()

    # 3. Generar respuesta
    generated_text = ""
    with torch.no_grad():
        # Generamos token a token, pero con una condición de salida dinámica
        for _ in range(200): # Máximo 200 chars por respuesta para que no se enrolle
            # Recortar contexto si es muy largo (sliding window)
            idx_cond = context[:, -block_size:]
            
            # Predicción
            logits, _ = m(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Decodificar el token nuevo
            char = decode(idx_next[0].tolist())
            
            # --- LÓGICA DE PARADA ---
            # Si el modelo intenta empezar otro usuario o terminar, cortamos.
            # Nota: Al ser tokenización por caracteres, detectar "<|fin|>" es un poco más truco.
            # Simplemente imprimimos y acumulamos.
            
            print(char, end='')
            sys.stdout.flush()
            generated_text += char
            
            # Actualizamos contexto para el siguiente carácter
            context = torch.cat((context, idx_next), dim=1)
            
            # Si detectamos que ha escrito el token de fin, rompemos el bucle
            if TOKEN_END in generated_text:
                break
            # Si empieza a escribir el turno del usuario, cortamos también por seguridad
            if "<|usuario|>" in generated_text: 
                break