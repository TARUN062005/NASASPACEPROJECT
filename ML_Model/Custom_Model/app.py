from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import uuid
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Directory to store user models
MODEL_DIR = os.getenv('MODEL_DIR', 'user_models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Mock classes to replace the missing imports
class CustomDataPreprocessor:
    """Mock data preprocessor"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_column = None
        self.numeric_features = []
        self.categorical_features = []
        self.label_encoder = MockLabelEncoder()
    
    def preprocess_data(self, df, target_column):
        """Mock preprocessing"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]
        
        # Simple type detection
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()
        
        # Simple label encoding for categorical target
        if not pd.api.types.is_numeric_dtype(y):
            self.label_encoder.fit(y)
            y = self.label_encoder.transform(y)
        
        return X, y
    
    def preprocess_single_sample(self, sample):
        """Mock single sample preprocessing"""
        processed = []
        for feature in self.feature_columns:
            if feature in sample:
                value = sample[feature]
                # Simple numeric conversion
                if feature in self.numeric_features:
                    try:
                        processed.append(float(value))
                    except (ValueError, TypeError):
                        processed.append(0.0)
                else:
                    processed.append(str(value) if value is not None else "")
            else:
                # Default value for missing features
                if feature in self.numeric_features:
                    processed.append(0.0)
                else:
                    processed.append("")
        return np.array([processed])
    
    def get_preprocessor_summary(self):
        """Get preprocessor summary"""
        return {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'num_features': len(self.feature_columns)
        }

class CustomModel:
    """Mock custom model"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_trained = False
    
    def create_model(self, model_type='ensemble', training_params=None):
        """Mock model creation"""
        self.model_type = model_type
        self.model = MockModel()
        self.is_trained = False
    
    def train(self, X, y):
        """Mock training"""
        if len(X) == 0:
            raise ValueError("No data available for training")
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """Mock prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X), self.model.predict_proba(X)
    
    def get_model_summary(self):
        """Get model summary"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat()
        }

class MockModel:
    """Simple mock model for demonstration"""
    
    def __init__(self):
        self.classes_ = [0, 1]
    
    def fit(self, X, y):
        """Mock training"""
        pass
    
    def predict(self, X):
        """Mock prediction - returns random class"""
        return np.random.randint(0, 2, size=(len(X),))
    
    def predict_proba(self, X):
        """Mock probability prediction"""
        probas = np.random.rand(len(X), 2)
        return probas / probas.sum(axis=1, keepdims=True)

class MockLabelEncoder:
    """Mock label encoder"""
    
    def __init__(self):
        self.classes_ = []
    
    def fit(self, y):
        """Fit label encoder"""
        self.classes_ = sorted(y.unique())
        return self
    
    def transform(self, y):
        """Transform labels"""
        return y.map({cls: i for i, cls in enumerate(self.classes_)})
    
    def inverse_transform(self, y):
        """Inverse transform labels"""
        return [self.classes_[i] for i in y]

class UserModelManager:
    """Manages user models by saving/loading from file system."""
    
    @staticmethod
    def get_user_model_paths(user_id):
        """Get file paths for user model and preprocessor"""
        if not user_id or not isinstance(user_id, str):
            raise ValueError("Invalid user ID")
        
        model_path = os.path.join(MODEL_DIR, f"{user_id}_model.joblib")
        preprocessor_path = os.path.join(MODEL_DIR, f"{user_id}_preprocessor.joblib")
        return model_path, preprocessor_path
    
    @staticmethod
    def get_user_model(user_id):
        """Get user's model from storage"""
        try:
            model_path, preprocessor_path = UserModelManager.get_user_model_paths(user_id)
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                model = joblib.load(model_path)
                preprocessor = joblib.load(preprocessor_path)
                return model, preprocessor
            return None, None
        except Exception as e:
            print(f"Error loading model for user {user_id}: {e}")
            return None, None
    
    @staticmethod
    def set_user_model(user_id, model, preprocessor):
        """Save user's current model to storage"""
        try:
            if not user_id:
                raise ValueError("User ID is required")
            
            model_path, preprocessor_path = UserModelManager.get_user_model_paths(user_id)
            joblib.dump(model, model_path)
            joblib.dump(preprocessor, preprocessor_path)
            print(f"✅ Model and preprocessor saved for user {user_id}.")
        except Exception as e:
            print(f"Error saving model for user {user_id}: {e}")
            raise
    
    @staticmethod
    def delete_user_model(user_id):
        """Delete user's model from storage"""
        try:
            model_path, preprocessor_path = UserModelManager.get_user_model_paths(user_id)
            
            deleted_files = []
            if os.path.exists(model_path):
                os.remove(model_path)
                deleted_files.append('model')
            
            if os.path.exists(preprocessor_path):
                os.remove(preprocessor_path)
                deleted_files.append('preprocessor')
            
            print(f"🗑️ Deleted {', '.join(deleted_files)} for user {user_id}.")
            return len(deleted_files) > 0
        except Exception as e:
            print(f"Error deleting model for user {user_id}: {e}")
            return False
    
    @staticmethod
    def get_user_model_info(user_id):
        """Get user's model information"""
        try:
            model, preprocessor = UserModelManager.get_user_model(user_id)
            
            if model and preprocessor:
                return {
                    'has_model': True,
                    'model_info': model.get_model_summary(),
                    'preprocessor_info': preprocessor.get_preprocessor_summary()
                }
            else:
                return {'has_model': False}
        except Exception as e:
            print(f"Error getting model info for user {user_id}: {e}")
            return {'has_model': False}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        active_models = []
        if os.path.exists(MODEL_DIR):
            active_models = [name for name in os.listdir(MODEL_DIR) if name.endswith('_model.joblib')]
        
        return jsonify({
            'status': 'healthy',
            'service': 'Custom Model',
            'active_users': len(active_models),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a custom model for a user"""
    try:
        # Get form data
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        target_column = request.form.get('target_column')
        if not target_column:
            return jsonify({'error': 'Target column is required'}), 400
        
        model_type = request.form.get('model_type', 'ensemble')
        training_params_str = request.form.get('training_params', '{}')
        
        # Safely evaluate training parameters
        try:
            training_params = eval(training_params_str)
        except:
            training_params = {}
        
        print(f"🎯 Training custom model for user {user_id}")
        print(f"📊 Model type: {model_type}")
        print(f"🎯 Target column: {target_column}")
        print(f"📦 File: {file.filename}")
        
        # Read file based on extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            else:
                return jsonify({'error': f'Unsupported file type: {file_extension}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Validate dataset
        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Dataset too small. Minimum 10 samples required.'}), 400
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in dataset'}), 400
        
        print(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Preprocess data
        preprocessor = CustomDataPreprocessor()
        X, y = preprocessor.preprocess_data(df, target_column)
        
        # Create and train model
        model = CustomModel()
        model.create_model(model_type, training_params)
        model.train(X, y)
        
        # Save model
        UserModelManager.set_user_model(user_id, model, preprocessor)
        
        # Get model info
        model_info = UserModelManager.get_user_model_info(user_id)
        
        return jsonify({
            'success': True,
            'message': 'Custom model trained successfully',
            'user_id': user_id,
            'model_info': model_info,
            'dataset_info': {
                'original_shape': df.shape,
                'training_samples': len(X),
                'num_features': X.shape[1],
                'num_classes': len(np.unique(y)) if hasattr(y, '__len__') else 1
            }
        })
        
    except Exception as e:
        print(f"❌ Custom model training error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using user's custom model"""
    try:
        user_id = request.headers.get('X-User-ID') or request.json.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        model, preprocessor = UserModelManager.get_user_model(user_id)
        
        if not model or not preprocessor:
            return jsonify({'error': 'No trained model found for user. Please train a model first.'}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if isinstance(data, dict):
            samples = [data]
            is_batch = False
        elif isinstance(data, list):
            samples = data
            is_batch = True
        else:
            return jsonify({'error': 'Invalid data format. Expected object or array.'}), 400
        
        predictions = []
        
        for sample in samples:
            try:
                processed_sample = preprocessor.preprocess_single_sample(sample)
                
                pred_class, probabilities = model.predict(processed_sample)
                
                class_idx = pred_class[0]
                class_name = preprocessor.label_encoder.inverse_transform([class_idx])[0]
                confidence = float(np.max(probabilities[0])) if probabilities is not None else 1.0
                
                class_probabilities = {}
                if probabilities is not None:
                    for i, class_label in enumerate(preprocessor.label_encoder.classes_):
                        class_probabilities[class_label] = float(probabilities[0][i])
                
                explanation = get_custom_prediction_explanation(class_name, confidence, sample)
                
                predictions.append({
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'probabilities': class_probabilities,
                    'explanation': explanation,
                    'input_features': sample
                })
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'input_features': sample
                })
        
        response_data = {
            'success': True,
            'user_id': user_id,
            'predictions': predictions
        }
        
        if not is_batch:
            response_data['prediction'] = predictions[0]
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Custom model prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get information about user's current model"""
    try:
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        model_info = UserModelManager.get_user_model_info(user_id)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            **model_info
        })
        
    except Exception as e:
        print(f"❌ Get model info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/delete', methods=['DELETE'])
def delete_model():
    """Delete user's current model"""
    try:
        user_id = request.headers.get('X-User-ID') or request.json.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        success = UserModelManager.delete_user_model(user_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model deleted successfully',
                'user_id': user_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No model found to delete',
                'user_id': user_id
            })
        
    except Exception as e:
        print(f"❌ Delete model error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_expected_features():
    """Get the expected features for user's model"""
    try:
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        _, preprocessor = UserModelManager.get_user_model(user_id)
        
        if not preprocessor:
            return jsonify({'error': 'No trained model found for user'}), 400
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'expected_features': preprocessor.feature_columns,
            'target_column': preprocessor.target_column,
            'feature_types': {
                'numeric': preprocessor.numeric_features,
                'categorical': preprocessor.categorical_features
            }
        })
        
    except Exception as e:
        print(f"❌ Get features error: {e}")
        return jsonify({'error': str(e)}), 500

def get_custom_prediction_explanation(predicted_class, confidence, input_features):
    """Generate human-readable explanation for custom model prediction"""
    base_explanation = f"Predicted as '{predicted_class}' with {confidence:.1%} confidence."
    insights = []
    
    # Extract numeric features for insights
    numeric_features = {}
    for k, v in input_features.items():
        if isinstance(v, (int, float)):
            numeric_features[k] = v
    
    if numeric_features:
        for feature, value in list(numeric_features.items())[:3]:
            if value > 1000:
                insights.append(f"High {feature} ({value})")
            elif value < 0.1:
                insights.append(f"Very low {feature} ({value})")
    
    if insights:
        return base_explanation + " Key factors: " + ", ".join(insights) + "."
    else:
        return base_explanation + " Based on the trained custom model."

@app.route('/supported_formats', methods=['GET'])
def get_supported_formats():
    """Get supported file formats for training"""
    return jsonify({
        'success': True,
        'supported_formats': ['.csv', '.xlsx', '.xls'],
        'max_file_size': '10MB',
        'requirements': {
            'min_samples': 10,
            'min_features': 1,
            'max_features': 100
        }
    })

if __name__ == '__main__':
    port = int(os.getenv('CUSTOM_MODEL_PORT', 5004))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Custom Model Server on port {port}")
    print(f"💾 Model storage directory: {os.path.abspath(MODEL_DIR)}")
    
    model_count = 0
    if os.path.exists(MODEL_DIR):
        model_count = len([name for name in os.listdir(MODEL_DIR) if name.endswith('_model.joblib')])
    
    print(f"👥 Current models in storage: {model_count}")
    app.run(host='0.0.0.0', port=port, debug=debug)