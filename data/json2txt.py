import json
import os

# 1. OBTIENE LA RUTA DEL DIRECTORIO ACTUAL
# Esto asegura que busque los archivos donde está el script, no desde donde lanzas la terminal
dir_actual = os.path.dirname(os.path.abspath(__file__))

# Rutas de los archivos (Input y Output)
archivo_json = os.path.join(dir_actual, 'data.json')
archivo_txt = os.path.join(dir_actual, 'datos_chat.txt')

print(f"--- Convirtiendo JSON a TXT para entrenamiento ---")
print(f"Leyendo: {archivo_json}")

# 2. DEFINICIÓN DE TOKENS ESPECIALES
# Estos son los separadores que el modelo aprenderá para distinguir turnos
TOKEN_USER = "\n<|usuario|>\n"
TOKEN_BOT = "\n<|sancho|>\n"
TOKEN_END = "\n<|fin|>\n"

try:
    # 3. LEER EL JSON
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraemos la lista de conversaciones (la clave "dataset")
    conversaciones = data.get("dataset", [])
    
    texto_formateado = ""
    conteo = 0

    # 4. PROCESAR CADA DIÁLOGO
    for item in conversaciones:
        pregunta = item.get("pregunta", "").strip()
        respuesta = item.get("respuesta", "").strip()

        if pregunta and respuesta:
            # Construimos el bloque: Usuario -> Pregunta -> Sancho -> Respuesta -> Fin
            bloque = f"{TOKEN_USER}{pregunta}{TOKEN_BOT}{respuesta}{TOKEN_END}"
            texto_formateado += bloque
            conteo += 1

    # 5. GUARDAR EL TXT
    if conteo > 0:
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write(texto_formateado)
        
        print(f"¡Éxito! Se han procesado {conteo} diálogos.")
        print(f"Archivo generado en: {archivo_txt}")
        
        # Mostramos una muestra de cómo quedó
        print("\n--- MUESTRA DEL CONTENIDO ---")
        print(texto_formateado[:300]) 
        print("...")
    else:
        print("Advertencia: No se encontraron diálogos válidos en el JSON.")

except FileNotFoundError:
    print(f"ERROR: No encuentro el archivo 'data.json' en {dir_actual}")
    print("Asegúrate de que el json y este script estén juntos.")
except json.JSONDecodeError:
    print("ERROR: El archivo 'data.json' tiene un formato inválido. Revisa las comas y llaves.")