from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# Add ML models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'KOI_Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'TOI_Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'K2_Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Custom_Model'))

app = Flask(__name__)
CORS(app)

# Import and initialize all models
try:
    from KOI_Model.app import app as koi_app
    from TOI_Model.app import app as toi_app
    from K2_Model.app import app as k2_app
    from Custom_Model.app import app as custom_app
    
    # Blueprint registration
    app.register_blueprint(koi_app, url_prefix='/koi')
    app.register_blueprint(toi_app, url_prefix='/toi')
    app.register_blueprint(k2_app, url_prefix='/k2')
    app.register_blueprint(custom_app, url_prefix='/custom')
    
    print("✅ All ML models loaded successfully!")
    
except Exception as e:
    print(f"⚠️ Error loading ML models: {e}")

@app.route('/')
def index():
    return jsonify({
        "message": "NASA ML Models API",
        "models": ["KOI", "TOI", "K2", "Custom"],
        "endpoints": {
            "koi": "/koi",
            "toi": "/toi", 
            "k2": "/k2",
            "custom": "/custom"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models": 4})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)