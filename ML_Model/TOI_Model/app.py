from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import TOIDataPreprocessor
import traceback
from dotenv import load_dotenv
import json
from datetime import datetime
import threading
import time
import webbrowser

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
label_encoder = None

# Performance tracker
class ModelPerformanceTracker:
    def __init__(self, tracker_file='model_performance.json'):
        self.tracker_file = tracker_file
        self.performance_data = self.load_performance_data()
        
    def load_performance_data(self):
        """Load existing performance data or create new"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize with empty structure
        return {
            'training_history': [],
            'live_predictions': [],
            'model_metrics': {
                'current_accuracy': 0,
                'best_accuracy': 0,
                'total_predictions': 0,
                'training_samples': 0,
                'last_trained': None
            },
            'class_distribution': {},
            'feature_importance': {}
        }
    
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")
    
    def record_training_session(self, X_train, y_train, X_test, y_test, evaluation_results, feature_names):
        """Record a training session with metrics"""
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': evaluation_results['accuracy'],
            'class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
            'feature_count': len(feature_names),
            'evaluation_metrics': evaluation_results
        }
        
        self.performance_data['training_history'].append(training_record)
        
        # Update model metrics
        self.performance_data['model_metrics']['current_accuracy'] = evaluation_results['accuracy']
        self.performance_data['model_metrics']['best_accuracy'] = max(
            self.performance_data['model_metrics']['best_accuracy'],
            evaluation_results['accuracy']
        )
        self.performance_data['model_metrics']['training_samples'] = len(X_train)
        self.performance_data['model_metrics']['last_trained'] = datetime.now().isoformat()
        
        self.save_performance_data()
        return training_record
    
    def record_live_prediction(self, prediction_data, actual_class=None, confidence=None):
        """Record a live prediction for continuous learning"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': prediction_data.get('predicted_class'),
            'confidence': prediction_data.get('confidence'),
            'actual_class': actual_class,
            'features_used': list(prediction_data.get('input_features', {}).keys()),
            'all_probabilities': prediction_data.get('probabilities', {})
        }
        
        self.performance_data['live_predictions'].append(prediction_record)
        self.performance_data['model_metrics']['total_predictions'] += 1
        
        # Update class distribution
        pred_class = prediction_data.get('predicted_class')
        if pred_class:
            if pred_class not in self.performance_data['class_distribution']:
                self.performance_data['class_distribution'][pred_class] = 0
            self.performance_data['class_distribution'][pred_class] += 1
        
        self.save_performance_data()
        return prediction_record
    
    def get_performance_summary(self):
        """Get current performance summary"""
        metrics = self.performance_data['model_metrics']
        training_history = self.performance_data['training_history']
        
        summary = {
            'current_accuracy': metrics['current_accuracy'],
            'best_accuracy': metrics['best_accuracy'],
            'total_predictions': metrics['total_predictions'],
            'training_samples': metrics['training_samples'],
            'training_sessions': len(training_history),
            'class_distribution': self.performance_data['class_distribution'],
            'last_trained': metrics['last_trained']
        }
        
        # Add trend information
        if len(training_history) >= 2:
            recent_acc = training_history[-1]['accuracy']
            previous_acc = training_history[-2]['accuracy']
            summary['accuracy_trend'] = 'improving' if recent_acc > previous_acc else 'declining' if recent_acc < previous_acc else 'stable'
        else:
            summary['accuracy_trend'] = 'unknown'
        
        return summary
    
    def generate_performance_charts(self):
        """Generate comprehensive performance charts"""
        charts = {}
        
        try:
            # 1. Accuracy Progress Chart
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Accuracy over training sessions
            plt.subplot(2, 2, 1)
            training_history = self.performance_data['training_history']
            if training_history:
                sessions = range(1, len(training_history) + 1)
                accuracies = [session['accuracy'] for session in training_history]
                training_sizes = [session['training_samples'] for session in training_history]
                
                plt.plot(sessions, accuracies, 'b-o', linewidth=2, markersize=6)
                plt.xlabel('Training Session')
                plt.ylabel('Accuracy')
                plt.title('Model Accuracy Progress')
                plt.grid(True, alpha=0.3)
                
                # Annotate points with accuracy values
                for i, (session, acc) in enumerate(zip(sessions, accuracies)):
                    plt.annotate(f'{acc:.3f}', (session, acc), 
                               textcoords="offset points", xytext=(0,10), ha='center')
            
            # Subplot 2: Training Data Growth
            plt.subplot(2, 2, 2)
            if training_history:
                sessions = range(1, len(training_history) + 1)
                plt.plot(sessions, training_sizes, 'g-s', linewidth=2, markersize=6)
                plt.xlabel('Training Session')
                plt.ylabel('Training Samples')
                plt.title('Training Data Growth')
                plt.grid(True, alpha=0.3)
            
            # Subplot 3: Class Distribution
            plt.subplot(2, 2, 3)
            class_dist = self.performance_data['class_distribution']
            if class_dist:
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                plt.bar(classes, counts, color='lightcoral', edgecolor='darkred')
                plt.xlabel('Classes')
                plt.ylabel('Prediction Count')
                plt.title('Live Prediction Class Distribution')
                plt.xticks(rotation=45)
            
            # Subplot 4: Confidence Distribution
            plt.subplot(2, 2, 4)
            live_predictions = self.performance_data['live_predictions']
            if live_predictions:
                confidences = [pred['confidence'] for pred in live_predictions if pred.get('confidence')]
                if confidences:
                    plt.hist(confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
                    plt.xlabel('Confidence')
                    plt.ylabel('Frequency')
                    plt.title('Prediction Confidence Distribution')
                    plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
                    plt.legend()
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            charts['performance_dashboard'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # 2. Live Prediction Timeline
            if len(live_predictions) > 1:
                plt.figure(figsize=(10, 6))
                
                # Convert timestamps and get confidences
                timestamps = [datetime.fromisoformat(pred['timestamp']) for pred in live_predictions[-50:]]  # Last 50 predictions
                confidences = [pred.get('confidence', 0) for pred in live_predictions[-50:]]
                
                plt.plot(timestamps, confidences, 'o-', alpha=0.7)
                plt.xlabel('Time')
                plt.ylabel('Confidence')
                plt.title('Live Prediction Confidence Timeline')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                charts['confidence_timeline'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
            
            # 3. Model Metrics Gauge
            plt.figure(figsize=(8, 6))
            metrics = self.performance_data['model_metrics']
            
            # Create a simple metrics table
            metric_data = [
                ['Current Accuracy', f"{metrics['current_accuracy']:.3f}"],
                ['Best Accuracy', f"{metrics['best_accuracy']:.3f}"],
                ['Training Samples', f"{metrics['training_samples']}"],
                ['Total Predictions', f"{metrics['total_predictions']}"],
                ['Training Sessions', f"{len(training_history)}"]
            ]
            
            plt.axis('off')
            table = plt.table(cellText=metric_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.3, 0.8, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            plt.title('Model Performance Summary', fontsize=14, pad=20)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            charts['metrics_summary'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
        except Exception as e:
            print(f"Chart generation error: {e}")
        
        return charts
    
    def get_training_suggestions(self):
        """Provide suggestions for model improvement"""
        suggestions = []
        summary = self.get_performance_summary()
        
        if summary['training_sessions'] == 0:
            suggestions.append("🚀 Train your model with initial dataset to get started")
        
        if summary['current_accuracy'] < 0.7:
            suggestions.append("📊 Consider adding more diverse training data")
        
        if len(summary['class_distribution']) < 2:
            suggestions.append("🎯 Add predictions for different classes to improve model balance")
        
        if summary['total_predictions'] > 100 and summary['training_sessions'] == 1:
            suggestions.append("🔄 Consider retraining with new prediction data")
        
        if summary['training_samples'] < 1000:
            suggestions.append("📈 More training data would likely improve accuracy")
        
        return suggestions

# Global tracker instance
performance_tracker = ModelPerformanceTracker()

class TOIModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def create_advanced_model(self):
        """Create an advanced ensemble model"""
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create multiple models for ensemble
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            eval_metric='mlogloss',
            random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb),
                ('rf', rf),
                ('lr', lr)
            ],
            voting='soft',
            weights=[3, 2, 1]
        )
        
    def train(self, X, y):
        """Train the model"""
        print("🚀 Training advanced ensemble model...")
        self.create_advanced_model()
        self.model.fit(X, y)
        self.is_trained = True
        print("✅ Model training completed")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        y_pred, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    def save_model(self, file_path):
        """Save the trained model"""
        joblib.dump(self.model, file_path)
        print(f"✅ Model saved to {file_path}")
    
    def load_model(self, file_path):
        """Load a trained model"""
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"✅ Model loaded from {file_path}")

def initialize_model():
    """Initialize or load the model"""
    global model, preprocessor, label_encoder
    
    model_path = 'model.pkl'
    preprocessor_path = 'preprocessor.pkl'
    
    try:
        # Initialize preprocessor
        preprocessor = TOIDataPreprocessor()
        
        # Try to load existing model and preprocessor
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = TOIModel()
            model.load_model(model_path)
            preprocessor.load_preprocessor(preprocessor_path)
            label_encoder = preprocessor.label_encoder
            print("✅ Pre-trained model loaded successfully")
        else:
            model = TOIModel()
            print("ℹ️ No pre-trained model found. Train the model first.")
            
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        model = TOIModel()
        preprocessor = TOIDataPreprocessor()

# Initialize model when app starts
initialize_model()

def generate_prediction_charts(predicted_class, confidence, probabilities, input_features):
    """Generate charts and return as base64 images"""
    charts = {}
    
    try:
        # 1. Confidence Gauge Chart
        plt.figure(figsize=(8, 4))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Confidence gauge
        ax1.barh([0], [confidence], color='skyblue', edgecolor='navy')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Confidence')
        ax1.set_title(f'Confidence: {confidence:.1%}')
        ax1.grid(True, alpha=0.3)
        
        # Probability distribution
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        colors = ['lightcoral' if cls != predicted_class else 'lightgreen' for cls in classes]
        
        ax2.bar(classes, probs, color=colors, edgecolor='black')
        ax2.set_ylabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['confidence_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Feature Importance (simulated)
        plt.figure(figsize=(10, 6))
        if input_features:
            features = list(input_features.keys())[:8]  # Top 8 features
            values = list(input_features.values())[:8]
            
            # Normalize values for better visualization
            values_normalized = [abs(v) / max(abs(max(values)), 1) for v in values]
            
            plt.barh(features, values_normalized, color='lightseagreen', edgecolor='black')
            plt.xlabel('Normalized Value')
            plt.title('Input Feature Values')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            charts['feature_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
    except Exception as e:
        print(f"Chart generation error: {e}")
    
    return charts

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model.is_trained if model else False,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train the TOI model"""
    global model, preprocessor, label_encoder
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        file_path = f"temp_{file.filename}"
        file.save(file_path)
        
        # Initialize preprocessor
        preprocessor = TOIDataPreprocessor()
        
        # Preprocess data
        X, y = preprocessor.preprocess_pipeline(file_path)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = TOIModel()
        model.train(X_train, y_train)
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test)
        
        # Record training session
        feature_names = preprocessor.selected_features if preprocessor else [f'feature_{i}' for i in range(X_train.shape[1])]
        training_record = performance_tracker.record_training_session(
            X_train, y_train, X_test, y_test, evaluation, feature_names
        )
        
        # Save model and preprocessor
        model.save_model('model.pkl')
        preprocessor.save_preprocessor('preprocessor.pkl')
        label_encoder = preprocessor.label_encoder
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'evaluation': evaluation,
            'class_names': label_encoder.classes_.tolist(),
            'training_record': training_record
        })
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on single or multiple samples"""
    global model, preprocessor, label_encoder
    
    try:
        if not model or not model.is_trained:
            return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle both single sample and batch predictions
        if isinstance(data, dict):
            samples = [data]
            is_batch = False
        elif isinstance(data, list):
            samples = data
            is_batch = True
        else:
            return jsonify({'error': 'Invalid data format. Expected object or array.'}), 400
        
        predictions = []
        charts = None
        
        for sample in samples:
            try:
                # Preprocess sample
                processed_sample = preprocessor.preprocess_single_sample(sample)
                
                # Make prediction
                pred_class, probabilities = model.predict(processed_sample)
                
                # Get class name and confidence
                class_idx = pred_class[0]
                class_name = label_encoder.inverse_transform([class_idx])[0]
                confidence = float(np.max(probabilities[0]))
                
                # Get all class probabilities
                class_probabilities = {}
                for i, class_label in enumerate(label_encoder.classes_):
                    class_probabilities[class_label] = float(probabilities[0][i])
                
                # Create explanation
                explanation = get_prediction_explanation(class_name, confidence, sample)
                
                prediction_data = {
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'probabilities': class_probabilities,
                    'explanation': explanation,
                    'input_features': sample,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Record prediction for potential future training
                performance_tracker.record_live_prediction(prediction_data)
                
                # Generate charts only for single prediction
                if not is_batch:
                    charts = generate_prediction_charts(class_name, confidence, class_probabilities, sample)
                    prediction_data['charts'] = charts
                
                predictions.append(prediction_data)
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'input_features': sample,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
        
        response_data = {
            'success': True,
            'predictions': predictions,
            'is_batch': is_batch,
            'total_predictions': len(predictions),
            'model_type': 'TOI',
            'performance_tracker': {
                'total_recorded': len(performance_tracker.performance_data['live_predictions']),
                'can_retrain': len([p for p in performance_tracker.performance_data['live_predictions'] if p.get('actual_class')]) >= 10
            }
        }
        
        if not is_batch:
            response_data['prediction'] = predictions[0]
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the trained model"""
    global model, preprocessor, label_encoder
    
    if not model or not model.is_trained:
        return jsonify({'error': 'Model not trained'}), 400
    
    info = {
        'is_trained': model.is_trained,
        'feature_columns': preprocessor.feature_columns if preprocessor else [],
        'selected_features': preprocessor.selected_features if preprocessor else [],
        'class_names': label_encoder.classes_.tolist() if label_encoder else [],
        'preprocessor_available': preprocessor is not None,
        'model_type': 'TOI'
    }
    
    return jsonify(info)

# New Routes for Performance Tracking
@app.route('/model/performance', methods=['GET'])
def get_model_performance():
    """Get current model performance metrics and charts"""
    try:
        if not model or not model.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Get performance summary
        performance_summary = performance_tracker.get_performance_summary()
        
        # Generate charts
        charts = performance_tracker.generate_performance_charts()
        
        # Get suggestions
        suggestions = performance_tracker.get_training_suggestions()
        
        return jsonify({
            'success': True,
            'performance_summary': performance_summary,
            'charts': charts,
            'suggestions': suggestions,
            'total_training_sessions': len(performance_tracker.performance_data['training_history']),
            'total_live_predictions': len(performance_tracker.performance_data['live_predictions'])
        })
        
    except Exception as e:
        print(f"Performance data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/record_prediction', methods=['POST'])
def record_prediction_for_training():
    """Record a prediction for future model training"""
    try:
        data = request.get_json()
        
        if not data or 'prediction' not in data:
            return jsonify({'error': 'Prediction data required'}), 400
        
        prediction_data = data['prediction']
        actual_class = data.get('actual_class')  # Optional: if user provides correct label
        
        # Record the prediction
        record = performance_tracker.record_live_prediction(prediction_data, actual_class)
        
        return jsonify({
            'success': True,
            'message': 'Prediction recorded for training',
            'record_id': len(performance_tracker.performance_data['live_predictions']),
            'total_recorded_predictions': len(performance_tracker.performance_data['live_predictions'])
        })
        
    except Exception as e:
        print(f"Prediction recording error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/retrain_with_new_data', methods=['POST'])
def retrain_with_new_data():
    """Retrain model with accumulated prediction data"""
    global model, preprocessor
    
    try:
        if not model or not model.is_trained:
            return jsonify({'error': 'No base model to retrain'}), 400
        
        # Get recorded predictions that have actual classes
        live_predictions = performance_tracker.performance_data['live_predictions']
        labeled_predictions = [pred for pred in live_predictions if pred.get('actual_class')]
        
        if len(labeled_predictions) < 10:
            return jsonify({
                'error': f'Need at least 10 labeled predictions for retraining. Currently have {len(labeled_predictions)}'
            }), 400
        
        # Convert predictions to training data format
        new_features = []
        new_labels = []
        
        for pred in labeled_predictions:
            # Extract features from prediction record
            features = extract_features_from_prediction(pred)
            if features is not None:
                new_features.append(features)
                new_labels.append(pred['actual_class'])
        
        if len(new_features) == 0:
            return jsonify({'error': 'Could not extract features from predictions'}), 400
        
        # Combine with original training data or retrain from scratch
        # For simplicity, we'll retrain with the new data
        X_new = np.array(new_features)
        y_new = np.array(new_labels)
        
        # Encode labels if needed
        if preprocessor and preprocessor.label_encoder:
            y_encoded = preprocessor.label_encoder.transform(y_new)
        else:
            # Create new label encoder
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_new)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Retrain model
        model.train(X_train, y_train)
        
        # Evaluate
        evaluation = model.evaluate(X_test, y_test)
        
        # Record training session
        feature_names = preprocessor.selected_features if preprocessor else [f'feature_{i}' for i in range(X_train.shape[1])]
        training_record = performance_tracker.record_training_session(
            X_train, y_train, X_test, y_test, evaluation, feature_names
        )
        
        # Save updated model
        model.save_model('model.pkl')
        
        return jsonify({
            'success': True,
            'message': f'Model retrained with {len(new_features)} new samples',
            'evaluation': evaluation,
            'training_record': training_record
        })
        
    except Exception as e:
        print(f"Retraining error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def extract_features_from_prediction(prediction_record):
    """Extract features from prediction record for retraining"""
    try:
        # This is a simplified example - adapt based on your actual feature structure
        features = []
        
        # Extract numeric features from input_features
        input_features = prediction_record.get('input_features', {})
        numeric_features = [
            'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade',
            'pl_insol', 'pl_eqt', 'st_tmag', 'st_dist',
            'st_teff', 'st_logg', 'st_rad'
        ]
        
        for feature in numeric_features:
            if feature in input_features:
                features.append(float(input_features[feature]))
            else:
                features.append(0.0)  # Default value
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def get_prediction_explanation(predicted_class, confidence, input_features):
    """Generate human-readable explanation for prediction"""
    explanations = {
        'FP': 'False Positive - The signal is likely caused by instrumental noise, stellar variability, or other astrophysical false positives rather than a planetary transit.',
        'PC': 'Planetary Candidate - This shows strong signatures of a planetary transit. Further observations and validation are recommended to confirm planetary nature.',
        'KP': 'Known Planet - This object has been previously confirmed as an exoplanet through multiple validation methods.',
        'CP': 'Confirmed Planet - Independently validated and confirmed as a bona fide exoplanet.',
        'APC': 'Ambiguous Planetary Candidate - The signal shows some planetary characteristics but requires additional data for confirmation.',
        'FA': 'False Alarm - The signal is likely an instrumental artifact or data processing error.'
    }
    
    base_explanation = explanations.get(predicted_class, 'Classification completed based on transit characteristics.')
    
    # Add feature-based insights
    insights = []
    
    if 'pl_trandep' in input_features:
        depth = input_features['pl_trandep']
        if depth > 10000:
            insights.append("Very deep transit suggests large planetary radius or small host star.")
        elif depth < 100:
            insights.append("Shallow transit may indicate small planet or requires high precision detection.")
    
    if 'pl_orbper' in input_features:
        period = input_features['pl_orbper']
        if period < 1:
            insights.append("Ultra-short orbital period typical of hot planets close to their host stars.")
        elif period > 100:
            insights.append("Long orbital period suggests distant orbit from host star.")
    
    if 'pl_rade' in input_features:
        radius = input_features['pl_rade']
        if radius > 20:
            insights.append("Large planetary radius, potentially a gas giant.")
        elif radius < 2:
            insights.append("Small planetary radius, potentially rocky planet.")
    
    if insights:
        feature_insight = " Feature insights: " + " ".join(insights)
        return base_explanation + feature_insight
    else:
        return base_explanation

# Auto Visualizer
class AutoVisualizer:
    def __init__(self, app_port=5001, visualizer_port=5002):
        self.app_port = app_port
        self.visualizer_port = visualizer_port
        
    def start_visualizer(self):
        """Start the visualizer web interface"""
        from flask import Flask as VisualizerFlask
        visualizer_app = VisualizerFlask(__name__)
        
        @visualizer_app.route('/')
        def dashboard():
            """Main performance dashboard"""
            try:
                # Get performance data from main app
                import requests
                response = requests.get(f'http://localhost:{self.app_port}/model/performance')
                if response.status_code == 200:
                    data = response.json()
                    
                    # Generate HTML dashboard
                    html = self.generate_dashboard_html(data)
                    return html
                else:
                    return f"<h1>Error connecting to model server</h1><p>{response.text}</p>"
                    
            except Exception as e:
                return f"<h1>Visualizer Error</h1><p>{str(e)}</p>"
        
        print(f"📊 Starting Model Performance Visualizer on port {self.visualizer_port}...")
        threading.Thread(
            target=lambda: visualizer_app.run(
                host='0.0.0.0', 
                port=self.visualizer_port, 
                debug=False, 
                use_reloader=False
            )
        ).start()
        
        # Wait a moment for server to start, then open browser
        time.sleep(2)
        print(f"🌐 Opening dashboard at http://localhost:{self.visualizer_port}")
        webbrowser.open(f'http://localhost:{self.visualizer_port}')
    
    def generate_dashboard_html(self, data):
        """Generate HTML dashboard with performance metrics"""
        
        performance = data.get('performance_summary', {})
        charts = data.get('charts', {})
        suggestions = data.get('suggestions', [])
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TOI Model Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .suggestions {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .suggestion-item {{ padding: 10px; margin: 5px 0; background: #e3f2fd; border-left: 4px solid #2196f3; }}
                .trend-improving {{ color: #4caf50; }}
                .trend-declining {{ color: #f44336; }}
                .trend-stable {{ color: #ff9800; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🚀 TOI Model Performance Dashboard</h1>
                    <p>Real-time monitoring of your exoplanet classification model</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{performance.get('current_accuracy', 0):.3f}</div>
                        <div class="metric-label">Current Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance.get('best_accuracy', 0):.3f}</div>
                        <div class="metric-label">Best Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance.get('training_samples', 0)}</div>
                        <div class="metric-label">Training Samples</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{performance.get('total_predictions', 0)}</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {f'trend-{performance.get("accuracy_trend", "unknown")}'}">
                            {performance.get('accuracy_trend', 'unknown').title()}
                        </div>
                        <div class="metric-label">Accuracy Trend</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    {self.generate_chart_html(charts)}
                </div>
                
                <div class="suggestions">
                    <h3>💡 Training Suggestions</h3>
                    {"".join([f'<div class="suggestion-item">{suggestion}</div>' for suggestion in suggestions])}
                    {'' if suggestions else '<p>No suggestions at this time. Model is performing well!</p>'}
                </div>
                
                <div style="text-align: center; margin-top: 20px; color: #666;">
                    <p>Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Dashboard auto-refreshes every 30 seconds</p>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function() {{
                    window.location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def generate_chart_html(self, charts):
        """Generate HTML for charts"""
        chart_html = ""
        
        if 'performance_dashboard' in charts:
            chart_html += f"""
            <div class="chart-container">
                <h3>Model Performance Dashboard</h3>
                <img src="data:image/png;base64,{charts['performance_dashboard']}" style="width: 100%;" alt="Performance Dashboard">
            </div>
            """
        
        if 'confidence_timeline' in charts:
            chart_html += f"""
            <div class="chart-container">
                <h3>Prediction Confidence Timeline</h3>
                <img src="data:image/png;base64,{charts['confidence_timeline']}" style="width: 100%;" alt="Confidence Timeline">
            </div>
            """
        
        if 'metrics_summary' in charts:
            chart_html += f"""
            <div class="chart-container">
                <h3>Model Metrics Summary</h3>
                <img src="data:image/png;base64,{charts['metrics_summary']}" style="width: 100%;" alt="Metrics Summary">
            </div>
            """
        
        return chart_html
    
    def start_auto_monitor(self):
        """Start automatic monitoring"""
        self.start_visualizer()
        print("✅ Auto-visualizer started! Performance dashboard is now active.")

def start_auto_visualizer():
    """Function to start the visualizer"""
    auto_visualizer = AutoVisualizer()
    auto_visualizer.start_auto_monitor()

if __name__ == '__main__':
    port = int(os.getenv('TOI_MODEL_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting TOI Model Server on port {port}")
    print(f"📊 Model ready: {model.is_trained if model else False}")
    
    # Start auto visualizer
    start_auto_visualizer()
    
    app.run(host='0.0.0.0', port=port, debug=debug)