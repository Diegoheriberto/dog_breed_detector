from flask import Flask, request, jsonify
import requests
import os
import random

app = Flask(__name__)

# Frases divertidas si no detecta raza animal
frases_divertidas = [
    "¡Pareces un rottweiler con lentes de sol!",
    "Hmm... eso se parece más a un humano disfrazado de chihuahua.",
    "¿Un husky? ¡No, un humano peludo!",
    "¡Parece un gato con autoestima de pastor alemán!",
    "Claramente eres... un golden retriever en cuerpo de humano 😄"
]

@app.route("/")
def home():
    return "Backend IA funcionando"

@app.route("/predict_breed", methods=["POST"])
def predict_breed():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No se proporcionó una URL de imagen"}), 400

    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/octet-stream"
    }

    try:
        # Descargar la imagen desde la URL
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content

        # Enviar imagen binaria al modelo
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/resnet-50",
            headers=headers,
            data=image_bytes
        )
        response.raise_for_status()
        predictions = response.json()

        if isinstance(predictions, list) and len(predictions) > 0:
            # Extraer top 3
            top3 = predictions[:3]
            razas_detectadas = [p["label"].lower() for p in top3]

            if any("dog" in r or "cat" in r for r in razas_detectadas):
                return jsonify({
                    "modo": "serio",
                    "predicciones": [
                        {"raza": p["label"], "confianza": round(p["score"] * 100, 2)}
                        for p in top3
                    ]
                })
            else:
                return jsonify({
                    "modo": "broma",
                    "mensaje": random.choice(frases_divertidas)
                })

        return jsonify({"error": "No se pudo obtener una predicción válida"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

