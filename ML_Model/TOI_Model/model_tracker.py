import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

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
        
        # Update class distribution based on predicted class
        pred_class = prediction_data.get('predicted_class')
        if pred_class:
            current_dist = self.performance_data['class_distribution']
            current_dist[pred_class] = current_dist.get(pred_class, 0) + 1
        
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
            # 1. Class Distribution Bar Chart (Improved)
            plt.figure(figsize=(8, 6))
            class_dist = self.performance_data.get('class_distribution', {})
            if class_dist:
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                # Order classes for consistency (e.g., alphabetically or by count)
                sorted_classes = sorted(class_dist, key=class_dist.get, reverse=True)
                sorted_counts = [class_dist[cls] for cls in sorted_classes]
                
                # Use a color palette
                colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_classes)))
                
                plt.bar(sorted_classes, sorted_counts, color=colors, edgecolor='black', zorder=2)
                
                plt.xlabel('Classes')
                plt.ylabel('Prediction Count')
                plt.title('Live Prediction Class Distribution')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
                
                # Annotate bars with counts
                for i, count in enumerate(sorted_counts):
                    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            charts['class_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # 2. Accuracy Progress Line Chart
            plt.figure(figsize=(8, 6))
            training_history = self.performance_data['training_history']
            if training_history:
                sessions = range(1, len(training_history) + 1)
                accuracies = [session['accuracy'] for session in training_history]
                
                plt.plot(sessions, accuracies, 'b-o', linewidth=2, markersize=6)
                plt.xlabel('Training Session')
                plt.ylabel('Accuracy')
                plt.title('Model Accuracy Progress')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                for i, (session, acc) in enumerate(zip(sessions, accuracies)):
                    plt.annotate(f'{acc:.3f}', (session, acc), 
                                 textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            charts['accuracy_progress'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # 3. Confidence Distribution Histogram
            plt.figure(figsize=(8, 6))
            live_predictions = self.performance_data['live_predictions']
            if live_predictions:
                confidences = [pred['confidence'] for pred in live_predictions if pred.get('confidence')]
                if confidences:
                    sns.histplot(confidences, bins=20, kde=True, color='purple', edgecolor='black', line_kws={'linestyle':'--'})
                    plt.xlabel('Confidence')
                    plt.ylabel('Frequency')
                    plt.title('Prediction Confidence Distribution')
                    plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
                    plt.legend()
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            charts['confidence_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # 4. Training Data Growth
            plt.figure(figsize=(8, 6))
            if training_history:
                sessions = range(1, len(training_history) + 1)
                training_sizes = [session['training_samples'] for session in training_history]
                plt.plot(sessions, training_sizes, 'g-s', linewidth=2, markersize=6)
                plt.xlabel('Training Session')
                plt.ylabel('Training Samples')
                plt.title('Training Data Growth')
                plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            charts['training_data_growth'] = base64.b64encode(buf.getvalue()).decode('utf-8')
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
        
        if len(summary.get('class_distribution', {})) < 2:
            suggestions.append("🎯 Add predictions for different classes to improve model balance")
        
        if summary['total_predictions'] > 100 and summary['training_sessions'] == 1:
            suggestions.append("🔄 Consider retraining with new prediction data")
        
        if summary['training_samples'] < 1000:
            suggestions.append("📈 More training data would likely improve accuracy")
        
        return suggestions

# Global tracker instance
performance_tracker = ModelPerformanceTracker()