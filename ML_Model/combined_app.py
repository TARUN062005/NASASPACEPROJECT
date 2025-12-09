from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        "message": "NASA ML Models API",
        "status": "running",
        "models": ["KOI", "TOI", "K2", "Custom"],
        "endpoints": {
            "health": "/health",
            "koi_predict": "/koi/predict (POST)",
            "toi_predict": "/toi/predict (POST)",
            "k2_predict": "/k2/predict (POST)",
            "custom_predict": "/custom/predict (POST)"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "NASA ML Models",
        "timestamp": "2025-12-09T18:00:00Z"
    })

# Mock prediction endpoints
@app.route('/koi/predict', methods=['POST'])
def koi_predict():
    return jsonify({
        "model": "KOI",
        "prediction": "CANDIDATE",
        "confidence": 0.78,
        "status": "success",
        "note": "Mock response - real model would be loaded here"
    })

@app.route('/toi/predict', methods=['POST'])
def toi_predict():
    return jsonify({
        "model": "TOI",
        "prediction": "CONFIRMED",
        "confidence": 0.85,
        "status": "success"
    })

@app.route('/k2/predict', methods=['POST'])
def k2_predict():
    return jsonify({
        "model": "K2",
        "prediction": "FALSE POSITIVE",
        "confidence": 0.67,
        "status": "success"
    })

@app.route('/custom/predict', methods=['POST'])
def custom_predict():
    return jsonify({
        "model": "Custom",
        "prediction": "CANDIDATE",
        "confidence": 0.92,
        "status": "success"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)