
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

app = Flask(__name__)
CORS(app)

modelo = MobileNetV2(weights='imagenet')

# Frases divertidas si no detecta raza animal
frases_divertidas = [
    "Esto es un misterio peludo. ¿Seguro que es una mascota? 🕵️‍♂️",
    "Ni con lupa encontramos la raza... ¡esto es arte abstracto peludo! 🎨",
    "La IA se rinde... dice que es un peluche con actitud. 🧸✨"
]

def procesar_imagen(ruta):
    img = image.load_img(ruta, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = modelo.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded

@app.route('/identificar', methods=['POST'])
def identificar():
    archivo = request.files.get('imagen')
    if not archivo:
        return jsonify({"error": "No se recibió imagen"}), 400

    os.makedirs("uploads", exist_ok=True)
    ruta = os.path.join("uploads", archivo.filename)
    archivo.save(ruta)

    try:
        predicciones = procesar_imagen(ruta)
        razas_detectadas = [p[1] for p in predicciones]
        confianza = float(predicciones[0][2]) * 100

        if any("dog" in r or "cat" in r for r in razas_detectadas):
            return jsonify({
                "modo": "serio",
                "predicciones": [
                    {
                        "raza": p[1],
                        "confianza": round(float(p[2]) * 100, 2)
                    } for p in predicciones
                ]
            })
        else:
            return jsonify({
                "modo": "broma",
                "mensaje": random.choice(frases_divertidas)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
