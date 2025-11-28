import torch
from sancho_model import GPTLanguageModel, device

# Cargar modelo
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('ckpt.pt', map_location=device))
model.eval()

# Crear una entrada falsa (Dummy input)
dummy_input = torch.randint(0, 50, (1, 64)).to(device) # (Batch 1, 64 tokens)

# Exportar
torch.onnx.export(model, dummy_input, "sancho_architecture.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch', 1: 'time'}, 'output': {0: 'batch'}})
print("¡Listo! Sube 'sancho_architecture.onnx' a netron.app")