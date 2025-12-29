import torch
import os
import json

# 1. IMPORTAMOS TU CÓDIGO (Asumiendo que tu archivo se llama sancho_gpt.py)
# Si tu archivo se llama de otra forma, cambia 'sancho_gpt' por el nombre de tu archivo.
import sancho_model as source_model 

# Definimos la carpeta de salida
output_dir = "sancho_hf_v1"
os.makedirs(output_dir, exist_ok=True)

print(f"--- Exportando modelo basado en: {source_model.__file__} ---")

# 2. INSTANCIAMOS EL MODELO
# Usamos las mismas variables que definiste en tu archivo para que las dimensiones cuadren
print("Creando instancia del modelo...")
model = source_model.GPTLanguageModel()

# 3. CARGAMOS LOS PESOS ENTRENADOS

directorio_script = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(directorio_script, 'ckpt.pt')
print(f"Cargando pesos desde {ckpt_path}...")
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# 4. GUARDAMOS LOS PESOS (Formato PyTorch estándar)
print("Guardando state_dict...")
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

# 5. GUARDAMOS LA CONFIGURACIÓN (Vital para saber las dimensiones)
# Hugging Face necesita un config.json para saber cómo construir el modelo
config = {
    "architectures": ["GPTLanguageModel"],
    "model_type": "custom_nano_gpt",
    "vocab_size": source_model.vocab_size,
    "n_embd": source_model.n_embd,
    "block_size": source_model.block_size,
    "n_head": source_model.n_head,
    "n_layer": source_model.n_layer,
    "dropout": source_model.dropout
}

with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# 6. GUARDAMOS EL VOCABULARIO (¡CRUCIAL!)
# Como tu tokenizador es por caracteres, necesitamos guardar la lista 'chars'.
# Sin esto, el modelo son solo números sin significado.
vocab_data = {
    "chars": source_model.chars,
    "stoi": source_model.stoi,
    "itos": source_model.itos
}

with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, indent=4, ensure_ascii=False)

print(f"¡Éxito! Modelo exportado en la carpeta '{output_dir}/'")
print("Archivos generados:")
print(f" - {output_dir}/pytorch_model.bin (Los pesos)")
print(f" - {output_dir}/config.json (Las dimensiones)")
print(f" - {output_dir}/vocab.json (Tu diccionario de caracteres)")