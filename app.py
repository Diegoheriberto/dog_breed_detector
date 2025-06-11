
import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo desde la carpeta model
MODEL_PATH = 'model/model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
else:
    print("❌ No se encontró el modelo")

# Lista de razas (ajústala a las que entrenaste)
razas = [
    "Golden Retriever",
    "Bulldog Francés",
    "Pastor Alemán",
    "Pug",
    "Beagle",
    "Chihuahua",
    "Labrador",
    "Shih Tzu",
    "Boxer",
    "Dálmata"
]

@app.route('/')
def home():
    return 'Backend de detección de razas funcionando 🐶'

@app.route('/predict_breed', methods=['POST'])
def predict_breed():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió archivo con clave "file"'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        indice = np.argmax(pred)
        raza = razas[indice]
        confianza = float(pred[indice])

        return jsonify({
            'raza': raza,
            'confianza': round(confianza, 3)
        })
    except Exception as e:
        logging.exception("Error al predecir la raza")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
