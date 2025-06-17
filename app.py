from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Backend IA funcionando"

@app.route("/predict_breed", methods=["POST"])
def predict_breed():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No se proporcionó una URL de imagen"}), 400

    api_token = os.getenv("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")
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    payload = {"inputs": image_url}

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/resnet-50",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        predictions = response.json()

        if isinstance(predictions, list) and len(predictions) > 0:
            top = predictions[0]
            return jsonify({
                "raza": top["label"],
                "confianza": round(top["score"] * 100, 2)
            })

        return jsonify({"error": "No se pudo obtener una predicción válida"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500