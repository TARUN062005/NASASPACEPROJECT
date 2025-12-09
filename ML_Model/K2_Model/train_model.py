# k2_train_model.py - FIXED VERSION
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
print(f"📊 K2 Model Trainer Starting...")

class K2DataPreprocessor:
    def __init__(self):
        # Updated to match the actual available features
        self.selected_features = [
            'pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt', 
            'st_teff', 'st_rad', 'st_logg', 'sy_vmag', 'sy_dist'
        ]
        self.target_column = 'disposition'
        self.feature_columns = self.selected_features
        self.label_encoder = None
        self.scaler = None
        
    def preprocess_pipeline(self, file_path):
        """Preprocess K2 dataset"""
        try:
            # Load data
            print(f"📁 Loading K2 data from {file_path}")
            
            try:
                df = pd.read_csv(file_path, comment='#', low_memory=False)
            except:
                df = pd.read_csv(file_path, low_memory=False)
            
            print(f"📊 K2 Dataset shape: {df.shape}")
            print(f"📊 Available columns: {list(df.columns)}")
            
            # Check target column names
            target_candidates = ['disposition', 'k2_disp', 'disp', 'class', 'target']
            target_column = None
            
            for candidate in target_candidates:
                if candidate in df.columns:
                    target_column = candidate
                    self.target_column = candidate
                    print(f"🎯 Using target column: {target_column}")
                    break
            
            if not target_column:
                print("⚠️ No target column found. Creating synthetic target...")
                df['disposition'] = self.create_synthetic_target(df)
                self.target_column = 'disposition'
            
            # Find available features from our selected list
            available_features = [f for f in self.selected_features if f in df.columns]
            print(f"🔍 Available features: {available_features}")
            
            if len(available_features) < 5:
                print("⚠️ Not enough features available. Adding alternative features...")
                # Add alternative features
                alternative_features = ['pl_bmasse', 'st_mass', 'st_met', 'pl_orbsmax']
                for alt in alternative_features:
                    if alt in df.columns and alt not in available_features:
                        available_features.append(alt)
            
            self.selected_features = available_features
            print(f"✅ Using features: {self.selected_features}")
            
            # Handle missing values
            print("🧹 Handling missing values...")
            
            # Keep only rows with essential features and target
            essential_features = self.selected_features + [self.target_column]
            df = df.dropna(subset=essential_features)
            print(f"✅ After dropping NAs: {df.shape[0]} samples")
            
            # Select features and target
            X = df[self.selected_features].copy()
            y = df[self.target_column].copy()
            
            # Clean target values
            y = y.astype(str).str.strip().str.upper()
            
            # Map to standard classes
            disposition_map = {
                'CONFIRMED': 'CONFIRMED',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE POSITIVE',
                'FP': 'FALSE POSITIVE',
                'PC': 'CANDIDATE',
                'KP': 'CONFIRMED',
                'CP': 'CONFIRMED',
                'FALSE POSITIVE (EB)': 'FALSE POSITIVE',
                'FALSE POSITIVE (VARIABLE)': 'FALSE POSITIVE',
                'NOT DISPOSITIONED': 'CANDIDATE'
            }
            
            y = y.map(disposition_map)
            
            # Filter to only common classes
            valid_classes = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
            mask = y.isin(valid_classes)
            y = y[mask]
            X = X.loc[y.index]
            
            # Check if we have enough samples
            if len(X) < 50:
                print(f"⚠️ Only {len(X)} samples after filtering. Augmenting with synthetic data...")
                X, y = self.augment_data(X, y, len(self.selected_features))
            
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
            
            print(f"✅ Preprocessed {len(X)} K2 samples with {len(self.selected_features)} features")
            return X_scaled, y_encoded
            
        except Exception as e:
            print(f"❌ K2 Preprocessing error: {e}")
            traceback.print_exc()
            return None, None
    
    def augment_data(self, X_real, y_real, n_features):
        """Augment data with synthetic samples"""
        n_samples = max(200 - len(X_real), 50)
        np.random.seed(42)
        
        # Generate synthetic features
        X_synth = np.random.randn(n_samples, n_features)
        
        # Add structure based on real data
        if len(X_real) > 0:
            for i in range(n_features):
                mean = np.mean(X_real[:, i]) if len(X_real) > 0 else 0
                std = np.std(X_real[:, i]) if len(X_real) > 0 else 1
                X_synth[:, i] = X_synth[:, i] * std + mean
        
        # Generate synthetic labels
        classes = np.unique(y_real)
        if len(classes) < 3:
            classes = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        
        y_synth = np.random.choice(classes, n_samples)
        
        # Encode if needed
        if hasattr(self.label_encoder, 'transform'):
            y_synth_encoded = self.label_encoder.transform(y_synth)
        else:
            self.label_encoder = LabelEncoder()
            y_synth_encoded = self.label_encoder.fit_transform(y_synth)
        
        # Combine
        X_combined = np.vstack([X_real, X_synth])
        y_combined = np.concatenate([y_real, y_synth_encoded])
        
        print(f"📈 Augmented data: {len(X_real)} real + {n_samples} synthetic = {len(X_combined)} total")
        return X_combined, y_combined
    
    def create_synthetic_target(self, df):
        """Create synthetic target for demonstration"""
        print("🤖 Creating synthetic K2 target based on features...")
        
        targets = []
        for idx, row in df.iterrows():
            score = 0
            
            if 'pl_rade' in df.columns and not pd.isna(row.get('pl_rade')):
                if 0.8 < row['pl_rade'] < 20:
                    score += 1
                elif row['pl_rade'] >= 20:
                    score -= 1
            
            if 'pl_orbper' in df.columns and not pd.isna(row.get('pl_orbper')):
                if 0.5 < row['pl_orbper'] < 100:
                    score += 1
            
            if 'st_teff' in df.columns and not pd.isna(row.get('st_teff')):
                if 3000 < row['st_teff'] < 6500:
                    score += 1
            
            if score >= 2:
                targets.append('CONFIRMED')
            elif score >= 1:
                targets.append('CANDIDATE')
            else:
                targets.append('FALSE POSITIVE')
        
        return targets
    
    def save_preprocessor(self, file_path):
        """Save preprocessor in a numpy-compatible format"""
        try:
            save_data = {
                'selected_features': self.selected_features,
                'target_column': self.target_column,
                'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'label_encoder_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None,
                'n_features': len(self.selected_features)
            }
            
            joblib.dump(save_data, file_path)
            print(f"✅ K2 Preprocessor saved to {file_path}")
        except Exception as e:
            print(f"❌ Save K2 preprocessor error: {e}")
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
            
            print(f"✅ K2 Preprocessor loaded from {file_path}")
        except Exception as e:
            print(f"❌ Load K2 preprocessor error: {e}")
            traceback.print_exc()

class K2Model:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_type = "K2 Mission Classifier"
        
    def create_simple_model(self):
        """Create a simpler, more stable model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        print("🤖 Created Random Forest model for K2")
        
    def create_advanced_model(self):
        """Create an advanced ensemble model"""
        try:
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
            print("🤖 Created advanced ensemble model for K2")
            
        except ImportError:
            print("⚠️ XGBoost not available, using simpler ensemble")
            self.create_simple_model()
        
    def train(self, X, y):
        """Train the model"""
        print("🚀 Training K2 Mission model...")
        print(f"📊 Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        if X.shape[0] < 1000:
            print("📊 Small K2 dataset detected, using simple model")
            self.create_simple_model()
        else:
            print("📊 Large K2 dataset detected, using advanced ensemble")
            self.create_advanced_model()
        
        self.model.fit(X, y)
        self.is_trained = True
        print("✅ K2 Model training completed")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("K2 Model not trained yet")
        
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
            save_data = {
                'model': self.model,
                'is_trained': self.is_trained,
                'model_type': self.model_type,
                'numpy_version': np.__version__,
                'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
            }
            
            joblib.dump(save_data, file_path, compress=3)
            print(f"✅ K2 Model saved to {file_path}")
            print(f"📊 NumPy version: {np.__version__}")
            print(f"📊 Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ Error saving K2 model: {e}")
            traceback.print_exc()
            raise
    
    def load_model(self, file_path):
        """Load a trained model"""
        try:
            saved_data = joblib.load(file_path)
            self.model = saved_data['model']
            self.is_trained = saved_data['is_trained']
            self.model_type = saved_data.get('model_type', 'Unknown')
            print(f"✅ K2 Model loaded from {file_path}")
            print(f"📊 NumPy version during save: {saved_data.get('numpy_version', 'Unknown')}")
            print(f"📊 Current NumPy version: {np.__version__}")
        except Exception as e:
            print(f"❌ Error loading K2 model: {e}")
            traceback.print_exc()
            raise

def train_k2_model(data_file_path=None):
    """Complete training pipeline for K2 model"""
    
    print("="*60)
    print("🎯 K2 Mission Model Training Pipeline")
    print("="*60)
    print(f"📊 Environment Info:")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print("="*60)
    
    try:
        if data_file_path and os.path.exists(data_file_path):
            print(f"📁 Using K2 data file: {data_file_path}")
        else:
            data_file_path = "k2_data.csv"
            if os.path.exists(data_file_path):
                print(f"📁 Found K2 data file: {data_file_path}")
            else:
                print("⚠️ No K2 data file found.")
                print("🤖 Creating synthetic K2 data for training...")
                return create_synthetic_training()
        
        preprocessor = K2DataPreprocessor()
        
        print("\n📊 Preprocessing K2 data...")
        X, y = preprocessor.preprocess_pipeline(data_file_path)
        
        if X is None or y is None:
            print("❌ Failed to preprocess K2 data.")
            print("🤖 Creating synthetic K2 data instead...")
            return create_synthetic_training()
        
        # Split data
        print("✂️ Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        print(f"📊 Features: {X_train.shape[1]}")
        
        # Train model
        print("\n🚀 Training K2 model...")
        model = K2Model()
        model.train(X_train, y_train)
        
        # Evaluate model
        print("\n📈 Evaluating K2 model performance...")
        evaluation = model.evaluate(X_test, y_test)
        
        # Save model and preprocessor
        print("\n💾 Saving K2 model and preprocessor...")
        model.save_model('k2_model.pkl')
        preprocessor.save_preprocessor('k2_preprocessor.pkl')
        
        # Print results
        print("\n" + "="*60)
        print("🎉 K2 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Accuracy: {evaluation['accuracy']:.4f}")
        print(f"🎯 Classes: {preprocessor.label_encoder.classes_.tolist()}")
        print(f"🔧 Features: {len(preprocessor.selected_features)}")
        print(f"📁 Model: k2_model.pkl")
        print(f"🔧 Preprocessor: k2_preprocessor.pkl")
        
        # Print classification summary
        print("\n📋 K2 Classification Summary:")
        class_report = evaluation['classification_report']
        
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in preprocessor.label_encoder.classes_:
            class_name_str = str(class_name)
            if class_name_str in class_report and class_name_str not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = class_report[class_name_str]
                print(f"{class_name_str:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10}")
        
        print(f"\n{'Accuracy':<20} {class_report['accuracy']:<10.3f}")
        print(f"{'Macro Avg':<20} {class_report['macro avg']['f1-score']:<10.3f}")
        print(f"{'Weighted Avg':<20} {class_report['weighted avg']['f1-score']:<10.3f}")
        
        return model, preprocessor, evaluation
        
    except Exception as e:
        print(f"\n❌ K2 Training failed: {e}")
        traceback.print_exc()
        return None, None, None

def create_synthetic_training():
    """Create synthetic K2 data and train model"""
    print("🤖 Creating synthetic K2 training data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = {
        'pl_orbper': np.random.exponential(20, n_samples),
        'pl_rade': np.random.uniform(0.5, 20, n_samples),
        'pl_insol': np.random.uniform(0.1, 1000, n_samples),
        'pl_eqt': np.random.uniform(300, 2000, n_samples),
        'st_teff': np.random.uniform(3000, 7000, n_samples),
        'st_rad': np.random.uniform(0.1, 10, n_samples),
        'st_logg': np.random.uniform(3.5, 5.0, n_samples),
        'sy_vmag': np.random.uniform(8, 16, n_samples),
        'sy_dist': np.random.uniform(10, 1000, n_samples),
        'disposition': []
    }
    
    # Create synthetic targets
    for i in range(n_samples):
        score = 0
        
        if 0.8 < synthetic_data['pl_rade'][i] < 20:
            score += 1
        if 0.5 < synthetic_data['pl_orbper'][i] < 100:
            score += 1
        if 3000 < synthetic_data['st_teff'][i] < 6500:
            score += 1
        
        if score >= 2:
            synthetic_data['disposition'].append('CONFIRMED')
        elif score >= 1:
            synthetic_data['disposition'].append('CANDIDATE')
        else:
            synthetic_data['disposition'].append('FALSE POSITIVE')
    
    df = pd.DataFrame(synthetic_data)
    df.to_csv('k2_synthetic_data.csv', index=False)
    print(f"💾 Saved synthetic K2 data to k2_synthetic_data.csv")
    
    # Train with synthetic data
    preprocessor = K2DataPreprocessor()
    
    X = df[preprocessor.selected_features].copy()
    y = df['disposition'].copy()
    
    preprocessor.label_encoder = LabelEncoder()
    y_encoded = preprocessor.label_encoder.fit_transform(y)
    
    preprocessor.scaler = StandardScaler()
    X_scaled = preprocessor.scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = K2Model()
    model.create_simple_model()
    model.model.fit(X_train, y_train)
    model.is_trained = True
    
    evaluation = model.evaluate(X_test, y_test)
    
    model.save_model('k2_model.pkl')
    preprocessor.save_preprocessor('k2_preprocessor.pkl')
    
    print(f"✅ Synthetic K2 model trained with accuracy: {evaluation['accuracy']:.4f}")
    return model, preprocessor, evaluation

def create_compatible_k2_server_files():
    """Create basic K2 server files for immediate use"""
    print("\n🔧 Creating K2 server compatibility files...")
    
    preprocessor = K2DataPreprocessor()
    model = K2Model()
    
    try:
        basic_preprocessor = {
            'selected_features': preprocessor.selected_features,
            'target_column': preprocessor.target_column,
            'scaler_mean': None,
            'scaler_scale': None,
            'label_encoder_classes': ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'],
            'numpy_version': np.__version__,
            'n_features': len(preprocessor.selected_features)
        }
        
        joblib.dump(basic_preprocessor, 'k2_preprocessor.pkl')
        print("✅ Created basic K2 preprocessor file")
        
        model_info = {
            'is_trained': False,
            'model_type': 'K2 Random Forest',
            'numpy_version': np.__version__,
            'message': 'Model not trained - use k2_train_model.py first',
            'n_features': len(preprocessor.selected_features)
        }
        
        joblib.dump(model_info, 'k2_model.pkl')
        print("✅ Created basic K2 model info file")
        
    except Exception as e:
        print(f"❌ Error creating K2 compatibility files: {e}")

if __name__ == "__main__":
    print("🤖 K2 Mission Model Trainer")
    print("="*60)
    
    create_compatible_k2_server_files()
    
    print("\n" + "="*60)
    print("🚀 Starting K2 Training Process")
    print("="*60)
    
    data_files = ['k2_data.csv', 'k2_train.csv', 'K2_data.csv', 'K2_train.csv']
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"📁 Found K2 data file: {data_file}")
            train_k2_model(data_file)
            break
    else:
        print("⚠️ No K2 data file found. Creating synthetic model...")
        train_k2_model()