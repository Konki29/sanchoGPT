import torch
import os
import time

# IMPORTANTE: Importa tu arquitectura original
import model.sancho_model as source_model 

# --- CONFIGURACIÓN OPTIMIZADA PARA RTX 5070 Ti ---
batch_size = 32       # Subimos de 4 a 32 (Tu gráfica va sobrada)
block_size = 256
max_iters = 5000       # Menos iteraciones porque el batch_size es mayor (aprende más rápido)
learning_rate = 1e-5  # Fine-tuning suave
eval_interval = 20
eval_iters = 40       # Cuántos batches usa para calcular el error promedio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Si tienes una serie 40/50, usar bfloat16 es más rápido y preciso que float16
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print(f"--- Usando dispositivo: {device} ---")
if device == 'cuda':
    print(f"Gráfica: {torch.cuda.get_device_name(0)}")

# --- 1. CARGAR DATOS DE CHAT ---
archivo_datos = os.path.join('data', 'datos_chat.txt')
if not os.path.exists(archivo_datos):
    print(f"ERROR: No encuentro {archivo_datos}. Ejecuta primero json2txt.py")
    exit()

with open(archivo_datos, 'r', encoding='utf-8') as f:
    text = f.read()

# --- VERIFICACIÓN DE VOCABULARIO (CRÍTICO) ---
stoi = source_model.stoi
itos = source_model.itos

# Comprobamos si los caracteres especiales de tu formato existen en el vocabulario base
chars_necesarios = set("<|>")
chars_faltantes = [c for c in chars_necesarios if c not in stoi]

if chars_faltantes:
    print(f"\n ADVERTENCIA CRÍTICA: Tu modelo base NO conoce estos caracteres: {chars_faltantes}")
    print("    El formato <|usuario|> se romperá porque se convertirán en espacios.")
    print("    SOLUCIÓN RECOMENDADA: Cambia los separadores en json2txt.py a caracteres que sí existan")
    print("    (ej: usa guiones o corchetes si existen, o simplemente palabras como USER: y SANCHO:)\n")
    # No paramos el script, pero ya sabes que puede fallar la estructura.

encode = lambda s: [stoi.get(c, stoi[' ']) for c in s] 
data = torch.tensor(encode(text), dtype=torch.long)

# Split train/val (90% entreno, 10% validación)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset cargado: {len(data)} caracteres.")

# --- 2. PREPARAR MODELO ---
print("Cargando cerebro de Sancho...")
model = source_model.GPTLanguageModel()
model.to(device)

# CARGAMOS LOS PESOS
checkpoint_path = os.path.join('model', 'ckpt.pt')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Pesos base cargados correctamente.")
else:
    print("ERROR: No encuentro model/ckpt.pt")
    exit()

# Optimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Función para obtener lotes
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Validar que hay suficientes datos
    if len(data) <= block_size:
        ix = torch.zeros((batch_size,), dtype=torch.long) # Fallback si hay poquísimos datos
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Función para estimar pérdida (IMPORTANTE PARA VER SI MEJORA)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 3. BUCLE DE ENTRENAMIENTO ---
print("\n--- Iniciando Fine-Tuning ---")
t0 = time.time()

for iter in range(max_iters):
    
    # Evaluación periódica
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Paso {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Early Stopping manual: Si el error sube en validación, nos estamos pasando
        if iter > 0 and losses['val'] > losses['train'] * 1.5:
            print(" Cuidado: El loss de validación está subiendo (Overfitting). Podrías parar ya.")

    # Entrenamiento
    xb, yb = get_batch('train')
    
    # Mixed Precision para tu RTX 5070 (Opcional, pero recomendado)
    # Como el modelo es simple, standard float32 también vuela, pero esto es más pro.
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

t1 = time.time()
print(f"\n¡Entrenamiento completado en {t1-t0:.2f} segundos!")

# Guardar
ruta_guardado = 'sancho_chat.pt'
torch.save(model.state_dict(), ruta_guardado)
print(f" Modelo guardado en: {ruta_guardado}")