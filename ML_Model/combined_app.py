from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import numpy as np
from datetime import datetime

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
        "timestamp": datetime.now().isoformat()
    })

# KOI endpoints
@app.route('/koi/health', methods=['GET'])
def koi_health():
    return jsonify({"status": "healthy", "model": "KOI"})

@app.route('/koi/predict', methods=['POST'])
def koi_predict():
    data = request.get_json()
    features = data.get('features', []) if data else []
    
    return jsonify({
        "model": "KOI",
        "prediction": "CANDIDATE",
        "confidence": 0.78,
        "features_received": len(features),
        "status": "success"
    })

# TOI endpoints
@app.route('/toi/health', methods=['GET'])
def toi_health():
    return jsonify({"status": "healthy", "model": "TOI"})

@app.route('/toi/predict', methods=['POST'])
def toi_predict():
    data = request.get_json()
    features = data.get('features', []) if data else []
    
    return jsonify({
        "model": "TOI",
        "prediction": "CONFIRMED",
        "confidence": 0.85,
        "features_received": len(features),
        "status": "success"
    })

# K2 endpoints
@app.route('/k2/health', methods=['GET'])
def k2_health():
    return jsonify({"status": "healthy", "model": "K2"})

@app.route('/k2/predict', methods=['POST'])
def k2_predict():
    data = request.get_json()
    features = data.get('features', []) if data else []
    
    return jsonify({
        "model": "K2",
        "prediction": "FALSE POSITIVE",
        "confidence": 0.67,
        "features_received": len(features),
        "status": "success"
    })

# Custom endpoints
@app.route('/custom/health', methods=['GET'])
def custom_health():
    return jsonify({"status": "healthy", "model": "Custom"})

@app.route('/custom/predict', methods=['POST'])
def custom_predict():
    data = request.get_json()
    features = data.get('features', []) if data else []
    
    return jsonify({
        "model": "Custom",
        "prediction": "CANDIDATE",
        "confidence": 0.92,
        "features_received": len(features),
        "status": "success"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 NASA ML Models API running on port {port}")
    app.run(host='0.0.0.0', port=port)