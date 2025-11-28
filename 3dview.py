import torch
from sancho_model import GPTLanguageModel, itos, device
import numpy as np

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('ckpt.pt', map_location=device))

# Guardar vectores
weights = model.token_embedding_table.weight.detach().cpu().numpy()
np.savetxt("vectors.tsv", weights, delimiter="\t")

# Guardar metadatos (SIN CABECERA)
with open("metadata.tsv", "w", encoding="utf-8") as f:
    # f.write("Character\n")  <--- ESTA LÍNEA SOBRABA, LA QUITAMOS
    for i in range(len(itos)):
        char = itos[i]
        # Escapar saltos de línea y tabuladores
        if char == "\n": char = "\\n"
        if char == "\t": char = "\\t"
        if char == " ": char = "[SPACE]"
        f.write(f"{char}\n")

print("Archivos vectors.tsv y metadata.tsv generados (ahora con 70 líneas cada uno).")