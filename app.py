
import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Límite de 10MB

# Cargar modelo
MODEL_PATH = 'model/model.h5'
if not os.path.exists(MODEL_PATH):
    logger.error("❌ No se encontró el modelo en la ruta especificada")
    sys.exit(1)

try:
    model = load_model(MODEL_PATH)
    logger.info("✅ Modelo cargado correctamente")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {str(e)}")
    sys.exit(1)

# Lista de razas
RAZAS = [
    "Golden Retriever", "Bulldog Francés", "Pastor Alemán",
    "Pug", "Beagle", "Chihuahua",
    "Labrador", "Shih Tzu", "Boxer", "Dálmata"
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_breed', methods=['POST'])
def predict_breed():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió archivo con clave "file"'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400

    try:
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        if file_length > 5 * 1024 * 1024:  # 5MB máximo
            return jsonify({'error': 'Archivo demasiado grande (máx 5MB)'}), 400

        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        indice = np.argmax(pred)

        if indice >= len(RAZAS):
            logger.warning(f"Índice {indice} fuera de rango")
            return jsonify({'error': 'Resultado fuera de rango'}), 500

        return jsonify({
            'raza': RAZAS[indice],
            'confianza': float(round(pred[indice], 3)),
            'distribucion': {RAZAS[i]: float(round(p, 3)) for i, p in enumerate(pred)}
        })

    except Exception as e:
        logger.exception("Error al predecir la raza")
        return jsonify({'error': 'Error procesando la imagen'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
