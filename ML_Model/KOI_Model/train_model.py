# train_model.py - Updated for numpy 1.26.0 compatibility
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import traceback
import warnings
warnings.filterwarnings('ignore')

print(f"📊 Using NumPy version: {np.__version__}")
print(f"📊 Using scikit-learn version: {joblib.__version__}")

class KOIDataPreprocessor:
    def __init__(self):
        self.selected_features = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
        ]
        self.target_column = 'koi_disposition'
        self.feature_columns = self.selected_features
        self.label_encoder = None
        self.scaler = None
        
    def preprocess_pipeline(self, file_path):
        """Preprocess KOI dataset"""
        try:
            # Load data
            print(f"📁 Loading data from {file_path}")
            df = pd.read_csv(file_path, comment='#', low_memory=False)
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📊 Sample columns: {list(df.columns)[:10]}...")
            
            # Handle missing values
            print("🧹 Handling missing values...")
            df = df.dropna(subset=self.selected_features + [self.target_column])
            print(f"✅ After dropping NAs: {df.shape[0]} samples")
            
            # Select features and target
            X = df[self.selected_features].copy()
            y = df[self.target_column].copy()
            
            # Filter to only common classes
            valid_classes = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'NOT DISPOSITIONED']
            y = y[y.isin(valid_classes)]
            X = X.loc[y.index]
            
            # Encode target variable
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            print(f"🎯 Classes: {self.label_encoder.classes_}")
            print(f"🎯 Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
            
            # Fill remaining missing values
            X = X.fillna(X.median())
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            print(f"✅ Preprocessed {len(X)} samples with {len(self.selected_features)} features")
            return X_scaled, y_encoded
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            traceback.print_exc()
            return None, None
    
    def save_preprocessor(self, file_path):
        """Save preprocessor in a numpy-compatible format"""
        try:
            # Save as a simple dictionary with basic types
            save_data = {
                'selected_features': self.selected_features,
                'target_column': self.target_column,
                'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'label_encoder_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None
            }
            
            joblib.dump(save_data, file_path)
            print(f"✅ Preprocessor saved to {file_path} (numpy {np.__version__} compatible)")
        except Exception as e:
            print(f"❌ Save preprocessor error: {e}")
            traceback.print_exc()
    
    def load_preprocessor(self, file_path):
        """Load preprocessor"""
        try:
            saved_data = joblib.load(file_path)
            self.selected_features = saved_data.get('selected_features', self.selected_features)
            self.target_column = saved_data.get('target_column', self.target_column)
            
            # Reconstruct scaler
            if 'scaler_mean' in saved_data and 'scaler_scale' in saved_data:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(saved_data['scaler_mean'])
                self.scaler.scale_ = np.array(saved_data['scaler_scale'])
                self.scaler.n_features_in_ = len(saved_data['scaler_mean'])
            
            # Reconstruct label encoder
            if 'label_encoder_classes' in saved_data:
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(saved_data['label_encoder_classes'])
            
            print(f"✅ Preprocessor loaded from {file_path}")
        except Exception as e:
            print(f"❌ Load preprocessor error: {e}")
            traceback.print_exc()

class KOIModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_type = "KOI Ensemble Classifier"
        
    def create_simple_model(self):
        """Create a simpler, more stable model"""
        # Use Random Forest only for better compatibility
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        print("🤖 Created Random Forest model for better compatibility")
        
    def create_advanced_model(self):
        """Create an advanced ensemble model"""
        try:
            # Try to use XGBoost, fall back if not available
            from xgboost import XGBClassifier
            
            xgb = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            lr = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            
            self.model = VotingClassifier(
                estimators=[
                    ('xgb', xgb),
                    ('rf', rf),
                    ('lr', lr)
                ],
                voting='soft',
                weights=[2, 2, 1]
            )
            print("🤖 Created advanced ensemble model")
            
        except ImportError:
            print("⚠️ XGBoost not available, using simpler ensemble")
            self.create_simple_model()
        
    def train(self, X, y):
        """Train the model"""
        print("🚀 Training KOI model...")
        print(f"📊 Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Create model based on data size
        if X.shape[0] < 1000:
            print("📊 Small dataset detected, using simple model")
            self.create_simple_model()
        else:
            print("📊 Large dataset detected, using advanced ensemble")
            self.create_advanced_model()
        
        self.model.fit(X, y)
        self.is_trained = True
        print("✅ Model training completed")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        else:
            probabilities = np.zeros((len(predictions), len(self.model.classes_)))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 0.8
                other_classes = [c for c in range(len(self.model.classes_)) if c != pred]
                probabilities[i, other_classes] = 0.2 / len(other_classes)
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist()
        }
    
    def save_model(self, file_path):
        """Save the trained model"""
        try:
            # Save model in a compatible format
            save_data = {
                'model': self.model,
                'is_trained': self.is_trained,
                'model_type': self.model_type,
                'numpy_version': np.__version__
            }
            
            joblib.dump(save_data, file_path, compress=3)
            print(f"✅ Model saved to {file_path}")
            print(f"📊 NumPy version: {np.__version__}")
            print(f"📊 Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            traceback.print_exc()
            raise
    
    def load_model(self, file_path):
        """Load a trained model"""
        try:
            saved_data = joblib.load(file_path)
            self.model = saved_data['model']
            self.is_trained = saved_data['is_trained']
            self.model_type = saved_data.get('model_type', 'Unknown')
            print(f"✅ Model loaded from {file_path}")
            print(f"📊 NumPy version during save: {saved_data.get('numpy_version', 'Unknown')}")
            print(f"📊 Current NumPy version: {np.__version__}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            raise

def train_koi_model(data_file_path=None):
    """Complete training pipeline for KOI model"""
    
    print("="*60)
    print("🎯 KOI Model Training Pipeline")
    print("="*60)
    print(f"📊 Environment Info:")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   scikit-learn: {joblib.__version__}")
    print("="*60)
    
    try:
        # Check for data file
        if data_file_path and os.path.exists(data_file_path):
            print(f"📁 Using data file: {data_file_path}")
        else:
            # Look for koi_data.csv in current directory
            data_file_path = "koi_data.csv"
            if os.path.exists(data_file_path):
                print(f"📁 Found data file: {data_file_path}")
            else:
                print("❌ No data file found. Please ensure 'koi_data.csv' exists.")
                return None, None, None
        
        # Initialize preprocessor
        preprocessor = KOIDataPreprocessor()
        
        # Preprocess data
        print("\n📊 Preprocessing data...")
        X, y = preprocessor.preprocess_pipeline(data_file_path)
        
        if X is None or y is None:
            print("❌ Failed to preprocess data. Exiting.")
            return None, None, None
        
        # Split data
        print("✂️ Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Train model
        print("\n🚀 Training model...")
        model = KOIModel()
        model.train(X_train, y_train)
        
        # Evaluate model
        print("\n📈 Evaluating model performance...")
        evaluation = model.evaluate(X_test, y_test)
        
        # Save model and preprocessor
        print("\n💾 Saving model and preprocessor...")
        model.save_model('koi_model.pkl')
        preprocessor.save_preprocessor('koi_preprocessor.pkl')
        
        # Print results
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Accuracy: {evaluation['accuracy']:.4f}")
        print(f"🎯 Classes: {preprocessor.label_encoder.classes_.tolist()}")
        print(f"🔧 Features: {len(preprocessor.selected_features)}")
        print(f"📁 Model: koi_model.pkl")
        print(f"🔧 Preprocessor: koi_preprocessor.pkl")
        
        # Print classification summary
        print("\n📋 Classification Summary:")
        class_report = evaluation['classification_report']
        
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in preprocessor.label_encoder.classes_:
            class_name_str = str(class_name)
            if class_name_str in class_report and class_name_str != 'accuracy' and class_name_str != 'macro avg' and class_name_str != 'weighted avg':
                metrics = class_report[class_name_str]
                print(f"{class_name_str:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10}")
        
        print(f"\n{'Accuracy':<20} {class_report['accuracy']:<10.3f}")
        print(f"{'Macro Avg':<20} {class_report['macro avg']['f1-score']:<10.3f}")
        print(f"{'Weighted Avg':<20} {class_report['weighted avg']['f1-score']:<10.3f}")
        
        return model, preprocessor, evaluation
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()
        return None, None, None

def create_compatible_server_files():
    """Create basic server files for immediate use"""
    print("\n🔧 Creating server compatibility files...")
    
    # Create a simple model and preprocessor
    preprocessor = KOIDataPreprocessor()
    model = KOIModel()
    
    # Save empty but compatible files
    try:
        # Save preprocessor with basic structure
        basic_preprocessor = {
            'selected_features': preprocessor.selected_features,
            'target_column': preprocessor.target_column,
            'scaler_mean': None,
            'scaler_scale': None,
            'label_encoder_classes': ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
            'numpy_version': np.__version__
        }
        
        joblib.dump(basic_preprocessor, 'koi_preprocessor.pkl')
        print("✅ Created basic preprocessor file")
        
        # Save model info file
        model_info = {
            'is_trained': False,
            'model_type': 'KOI Random Forest',
            'numpy_version': np.__version__,
            'message': 'Model not trained - use train_model.py first'
        }
        
        joblib.dump(model_info, 'koi_model.pkl')
        print("✅ Created basic model info file")
        
    except Exception as e:
        print(f"❌ Error creating compatibility files: {e}")

if __name__ == "__main__":
    print("🤖 KOI Model Trainer")
    print("="*60)
    
    # First, check if we can create compatible files
    create_compatible_server_files()
    
    # Then attempt training
    print("\n" + "="*60)
    print("🚀 Starting Training Process")
    print("="*60)
    
    train_koi_model()