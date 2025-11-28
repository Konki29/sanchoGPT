import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.sancho_model import GPTLanguageModel, device

# Cargar modelo
model = GPTLanguageModel().to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ckpt.pt')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Crear una entrada falsa (Dummy input)
dummy_input = torch.randint(0, 50, (1, 64)).to(device) # (Batch 1, 64 tokens)

# Exportar
torch.onnx.export(model, dummy_input, "sancho_architecture.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch', 1: 'time'}, 'output': {0: 'batch'}})
print("¡Listo! Sube 'sancho_architecture.onnx' a netron.app")