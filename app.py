from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 🔐 Coloca tu token Hugging Face real
HUGGINGFACE_API_TOKEN = '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'

@app.route('/')
def home():
    return 'API de predicción de raza funcionando 🐶'

@app.route('/predict_breed', methods=['POST'])
def predict_breed():
    data = request.get_json()
    image_url = data.get('url')

    if not image_url:
        return jsonify({'error': 'Falta la URL de imagen'}), 400

    try:
        # Descargamos la imagen desde la URL
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content

        # Enviamos la imagen binaria a Hugging Face
        response = requests.post(
            'https://api-inference.huggingface.co/models/microsoft/resnet-50',
            headers={
                'Authorization': f'Bearer {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}',
                'Content-Type': 'application/octet-stream'
            },
            data=image_bytes
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error al conectar con Hugging Face: {str(e)}'}), 500

    result = response.json()
    return jsonify({'predicciones': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)