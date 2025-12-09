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
import traceback

class KOIModelPerformanceTracker:
    def __init__(self, tracker_file='koi_model_performance.json'):
        self.tracker_file = tracker_file
        self.performance_data = self.load_performance_data()
        
    def load_performance_data(self):
        """Load existing performance data or create new"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading KOI performance data: {e}")
        
        # Initialize with KOI-specific structure
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
            'class_distribution': {
                'CONFIRMED': 0,
                'CANDIDATE': 0,
                'FALSE POSITIVE': 0,
                'NOT DISPOSITIONED': 0
            },
            'feature_importance': {
                'koi_period': 0.15,
                'koi_depth': 0.18,
                'koi_prad': 0.12,
                'koi_teq': 0.10,
                'koi_model_snr': 0.16,
                'koi_steff': 0.08,
                'koi_slogg': 0.07,
                'koi_srad': 0.06,
                'koi_impact': 0.05,
                'koi_duration': 0.03
            }
        }
    
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            print(f"✅ KOI Performance data saved to {self.tracker_file}")
        except Exception as e:
            print(f"❌ Error saving KOI performance data: {e}")
    
    def record_training_session(self, X_train, y_train, X_test, y_test, evaluation_results, feature_names):
        """Record a training session with metrics"""
        try:
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': float(evaluation_results['accuracy']),
                'class_distribution': self._safe_class_distribution(y_train),
                'feature_count': len(feature_names),
                'evaluation_metrics': evaluation_results
            }
            
            self.performance_data['training_history'].append(training_record)
            
            # Update model metrics
            self.performance_data['model_metrics']['current_accuracy'] = float(evaluation_results['accuracy'])
            self.performance_data['model_metrics']['best_accuracy'] = max(
                float(self.performance_data['model_metrics']['best_accuracy']),
                float(evaluation_results['accuracy'])
            )
            self.performance_data['model_metrics']['training_samples'] = len(X_train)
            self.performance_data['model_metrics']['last_trained'] = datetime.now().isoformat()
            
            # Update feature importance if available
            if 'feature_importance' in evaluation_results:
                self.performance_data['feature_importance'] = evaluation_results['feature_importance']
            
            self.save_performance_data()
            return training_record
            
        except Exception as e:
            print(f"❌ Error recording KOI training session: {e}")
            traceback.print_exc()
            return None
    
    def _safe_class_distribution(self, y):
        """Safely create class distribution dictionary"""
        try:
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique.astype(str), counts.astype(int)))
        except:
            return {'CONFIRMED': 0, 'CANDIDATE': 0, 'FALSE POSITIVE': 0, 'NOT DISPOSITIONED': 0}
    
    def record_live_prediction(self, prediction_data, actual_class=None, confidence=None):
        """Record a live prediction for continuous learning"""
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'predicted_class': prediction_data.get('predicted_class'),
                'confidence': float(prediction_data.get('confidence', 0)),
                'actual_class': actual_class,
                'features_used': list(prediction_data.get('input_features', {}).keys()),
                'all_probabilities': prediction_data.get('probabilities', {})
            }
            
            self.performance_data['live_predictions'].append(prediction_record)
            self.performance_data['model_metrics']['total_predictions'] += 1
            
            # Update class distribution
            pred_class = prediction_data.get('predicted_class')
            if pred_class and pred_class in self.performance_data['class_distribution']:
                self.performance_data['class_distribution'][pred_class] += 1
            
            self.save_performance_data()
            return prediction_record
            
        except Exception as e:
            print(f"❌ Error recording KOI live prediction: {e}")
            return None
    
    def get_performance_summary(self):
        """Get current performance summary"""
        try:
            metrics = self.performance_data['model_metrics']
            training_history = self.performance_data['training_history']
            
            summary = {
                'current_accuracy': float(metrics.get('current_accuracy', 0)),
                'best_accuracy': float(metrics.get('best_accuracy', 0)),
                'total_predictions': int(metrics.get('total_predictions', 0)),
                'training_samples': int(metrics.get('training_samples', 0)),
                'training_sessions': len(training_history),
                'class_distribution': self.performance_data.get('class_distribution', {}),
                'last_trained': metrics.get('last_trained')
            }
            
            # Add trend information
            if len(training_history) >= 2:
                recent_acc = training_history[-1].get('accuracy', 0)
                previous_acc = training_history[-2].get('accuracy', 0)
                summary['accuracy_trend'] = 'improving' if recent_acc > previous_acc else 'declining' if recent_acc < previous_acc else 'stable'
            else:
                summary['accuracy_trend'] = 'unknown'
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting KOI performance summary: {e}")
            return {
                'current_accuracy': 0,
                'best_accuracy': 0,
                'total_predictions': 0,
                'training_samples': 0,
                'training_sessions': 0,
                'class_distribution': {},
                'last_trained': None,
                'accuracy_trend': 'unknown'
            }
    
    def generate_performance_charts(self):
        """Generate comprehensive performance charts for KOI"""
        charts = {}
        
        try:
            # Set KOI-specific styling
            plt.style.use('dark_background')
            koi_colors = ['#8B5FBF', '#6A3093', '#9C27B0', '#E1BEE7']
            
            # 1. Accuracy Progress Chart
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Accuracy over training sessions
            plt.subplot(2, 2, 1)
            training_history = self.performance_data['training_history']
            if training_history and len(training_history) > 0:
                sessions = range(1, len(training_history) + 1)
                accuracies = [session.get('accuracy', 0) for session in training_history]
                
                plt.plot(sessions, accuracies, 'o-', linewidth=2, markersize=8, color=koi_colors[0])
                plt.fill_between(sessions, accuracies, alpha=0.3, color=koi_colors[0])
                plt.xlabel('Training Session')
                plt.ylabel('Accuracy')
                plt.title('KOI Model Accuracy Progress', fontweight='bold')
                plt.grid(True, alpha=0.2)
                
                # Annotate last point
                if len(accuracies) > 0:
                    plt.annotate(f'{accuracies[-1]:.3f}', 
                               (sessions[-1], accuracies[-1]),
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontweight='bold',
                               color='white')
            else:
                plt.text(0.5, 0.5, 'No training data yet', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('KOI Model Accuracy Progress', fontweight='bold')
            
            # Subplot 2: Class Distribution
            plt.subplot(2, 2, 2)
            class_dist = self.performance_data.get('class_distribution', {})
            if class_dist and any(count > 0 for count in class_dist.values()):
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                # KOI-specific colors for classes
                class_colors = {
                    'CONFIRMED': '#4CAF50',
                    'CANDIDATE': '#FFC107', 
                    'FALSE POSITIVE': '#F44336',
                    'NOT DISPOSITIONED': '#9E9E9E'
                }
                
                colors = [class_colors.get(cls, koi_colors[0]) for cls in classes]
                
                bars = plt.bar(classes, counts, color=colors, edgecolor='white', linewidth=1)
                plt.xlabel('KOI Classes')
                plt.ylabel('Prediction Count')
                plt.title('KOI Prediction Distribution', fontweight='bold')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No predictions yet', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('KOI Prediction Distribution', fontweight='bold')
            
            # Subplot 3: Feature Importance
            plt.subplot(2, 2, 3)
            feature_importance = self.performance_data.get('feature_importance', {})
            if feature_importance:
                features = list(feature_importance.keys())[:8]  # Top 8 features
                importance = list(feature_importance.values())[:8]
                
                # Sort by importance
                sorted_idx = np.argsort(importance)
                features = [features[i] for i in sorted_idx]
                importance = [importance[i] for i in sorted_idx]
                
                plt.barh(features, importance, color=koi_colors[1], edgecolor='white', linewidth=1)
                plt.xlabel('Importance Score')
                plt.title('KOI Feature Importance', fontweight='bold')
                
                # Add value labels
                for i, (feat, imp) in enumerate(zip(features, importance)):
                    plt.text(imp + 0.01, i, f'{imp:.2f}', va='center', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No feature data', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('KOI Feature Importance', fontweight='bold')
            
            # Subplot 4: Confidence Distribution
            plt.subplot(2, 2, 4)
            live_predictions = self.performance_data.get('live_predictions', [])
            if live_predictions:
                confidences = [pred.get('confidence', 0) for pred in live_predictions if pred.get('confidence')]
                if confidences:
                    plt.hist(confidences, bins=15, alpha=0.7, color=koi_colors[2], 
                            edgecolor='white', linewidth=1)
                    plt.xlabel('Confidence')
                    plt.ylabel('Frequency')
                    plt.title('KOI Confidence Distribution', fontweight='bold')
                    
                    mean_conf = np.mean(confidences)
                    plt.axvline(mean_conf, color='red', linestyle='--', 
                              label=f'Mean: {mean_conf:.3f}')
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, 'No confidence data', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('KOI Confidence Distribution', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No predictions yet', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('KOI Confidence Distribution', fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)
            charts['performance_dashboard'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # 2. Training Progress Timeline
            if len(training_history) > 1:
                plt.figure(figsize=(10, 6))
                
                timestamps = [datetime.fromisoformat(session['timestamp']) 
                            for session in training_history]
                accuracies = [session.get('accuracy', 0) for session in training_history]
                samples = [session.get('training_samples', 0) for session in training_history]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Accuracy timeline
                ax1.plot(timestamps, accuracies, 'o-', color=koi_colors[0], 
                        linewidth=2, markersize=6, label='Accuracy')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('KOI Training Accuracy Timeline', fontweight='bold')
                ax1.grid(True, alpha=0.2)
                ax1.legend()
                
                # Samples timeline
                ax2.plot(timestamps, samples, 's-', color=koi_colors[1], 
                        linewidth=2, markersize=4, label='Training Samples')
                ax2.set_ylabel('Samples')
                ax2.set_xlabel('Time')
                ax2.set_title('KOI Training Data Growth', fontweight='bold')
                ax2.grid(True, alpha=0.2)
                ax2.legend()
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                           facecolor='#1a1a1a', edgecolor='none')
                buf.seek(0)
                charts['training_timeline'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
            
            # 3. Model Metrics Summary
            plt.figure(figsize=(8, 6))
            summary = self.get_performance_summary()
            
            # Create metrics table
            metric_data = [
                ['Current Accuracy', f"{summary['current_accuracy']:.3f}"],
                ['Best Accuracy', f"{summary['best_accuracy']:.3f}"],
                ['Training Samples', f"{summary['training_samples']}"],
                ['Total Predictions', f"{summary['total_predictions']}"],
                ['Training Sessions', f"{summary['training_sessions']}"],
                ['Accuracy Trend', f"{summary['accuracy_trend'].title()}"]
            ]
            
            plt.axis('off')
            table = plt.table(cellText=metric_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.2, 0.8, 0.6])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            
            # Color cells based on values
            for i, (metric, value) in enumerate(metric_data):
                if 'accuracy' in metric.lower():
                    acc_value = float(value)
                    if acc_value > 0.8:
                        table[(i+1, 1)].set_facecolor('#4CAF50')
                    elif acc_value > 0.6:
                        table[(i+1, 1)].set_facecolor('#FFC107')
                    else:
                        table[(i+1, 1)].set_facecolor('#F44336')
            
            plt.title('KOI Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)
            charts['metrics_summary'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
        except Exception as e:
            print(f"❌ KOI Chart generation error: {e}")
            traceback.print_exc()
        
        return charts
    
    def get_training_suggestions(self):
        """Provide KOI-specific training suggestions"""
        suggestions = []
        summary = self.get_performance_summary()
        
        if summary['training_sessions'] == 0:
            suggestions.append("🚀 Train your KOI model with Kepler dataset to get started")
        
        if summary['current_accuracy'] < 0.7:
            suggestions.append("📊 Add more diverse Kepler data including different stellar types")
        
        class_dist = summary.get('class_distribution', {})
        if class_dist.get('NOT DISPOSITIONED', 0) > class_dist.get('CONFIRMED', 0):
            suggestions.append("🎯 Focus on collecting more CONFIRMED exoplanet data")
        
        if summary['total_predictions'] > 50 and summary['training_sessions'] == 1:
            suggestions.append("🔄 Retrain KOI model with new prediction data for improvement")
        
        if summary['training_samples'] < 1000:
            suggestions.append("📈 More training samples will improve KOI classification accuracy")
        
        # KOI-specific suggestions
        if not class_dist.get('CANDIDATE', 0):
            suggestions.append("🔭 Include more CANDIDATE class data for better planetary candidate detection")
        
        if summary['current_accuracy'] > 0.85:
            suggestions.append("⭐ KOI model performing well! Consider fine-tuning for specific planetary types")
        
        return suggestions

# Global KOI tracker instance
koi_performance_tracker = KOIModelPerformanceTracker()