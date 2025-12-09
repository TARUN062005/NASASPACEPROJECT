# k2_preprocess.py - Simplified K2 Data Preprocessor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class K2DataPreprocessor:
    def __init__(self):
        self.selected_features = [
            'pl_orbper', 'pl_rade', 'pl_trandep', 'pl_trandur',
            'pl_insol', 'pl_eqt', 'st_teff', 'st_rad',
            'st_logg', 'sy_vmag', 'sy_dist', 'k2_campaign'
        ]
        
        self.target_column = 'k2_disp'
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def preprocess_pipeline(self, file_path):
        """Main preprocessing pipeline for K2 data"""
        try:
            print(f"📁 Processing K2 data from {file_path}")
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
            
            print(f"📊 Original data shape: {df.shape}")
            
            # Check for required columns
            required_cols = ['pl_orbper', 'pl_rade', self.target_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
                print("🤖 Creating synthetic data for demonstration...")
                return self.create_synthetic_data()
            
            # Clean data
            df_clean = self.clean_data(df)
            print(f"✅ After cleaning: {df_clean.shape}")
            
            # Handle missing values
            df_filled = self.handle_missing_values(df_clean)
            
            # Feature engineering for K2
            df_features = self.k2_feature_engineering(df_filled)
            
            # Encode target
            X, y = self.prepare_features_target(df_features)
            
            print(f"🎯 Final features shape: {X.shape}")
            print(f"🎯 Target shape: {y.shape}")
            print(f"🎯 Feature columns: {len(self.feature_columns)}")
            print(f"🎯 Classes: {self.label_encoder.classes_}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def clean_data(self, df):
        """Clean K2 data"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Keep only rows with target values
        df = df[df[self.target_column].notna()]
        
        # Convert target to string and clean
        df[self.target_column] = df[self.target_column].astype(str).str.strip().str.upper()
        
        # Standardize K2 disposition values
        disposition_map = {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE POSITIVE',
            'FP': 'FALSE POSITIVE',
            'PC': 'CANDIDATE',
            'KP': 'CONFIRMED',
            'CP': 'CONFIRMED'
        }
        
        df[self.target_column] = df[self.target_column].map(disposition_map).fillna('FALSE POSITIVE')
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in K2 data"""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column and col in df.columns:
                if df[col].isna().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
        
        return df
    
    def k2_feature_engineering(self, df):
        """Create K2-specific features"""
        # Ensure we have campaign column
        if 'k2_campaign' not in df.columns:
            # Try to extract from other columns
            if 'kepid' in df.columns:
                # Simulate campaign based on ID
                df['k2_campaign'] = df['kepid'].apply(lambda x: f'C{(hash(str(x)) % 20)}')
            else:
                # Assign random campaigns C0-C19
                df['k2_campaign'] = [f'C{i%20}' for i in range(len(df))]
        
        # Encode campaign as numeric (one-hot would be better but we keep it simple)
        df['campaign_encoded'] = df['k2_campaign'].str.extract(r'C(\d+)').fillna(0).astype(int)
        
        # Calculate transit signal-to-noise ratio (simplified)
        if 'pl_trandep' in df.columns and 'pl_trandur' in df.columns:
            df['k2_snr'] = df['pl_trandep'] / (df['pl_trandur'] + 1)
        
        # Add stellar type based on temperature
        if 'st_teff' in df.columns:
            conditions = [
                df['st_teff'] < 3500,
                (df['st_teff'] >= 3500) & (df['st_teff'] < 5000),
                (df['st_teff'] >= 5000) & (df['st_teff'] < 6000),
                (df['st_teff'] >= 6000) & (df['st_teff'] < 7500),
                df['st_teff'] >= 7500
            ]
            choices = [0, 1, 2, 3, 4]  # M, K, G, F, A
            df['stellar_type'] = np.select(conditions, choices, default=2)
        
        return df
    
    def prepare_features_target(self, df):
        """Prepare features and target for training"""
        # Select features
        available_features = [f for f in self.selected_features if f in df.columns]
        
        # Add engineered features
        engineered_features = ['campaign_encoded', 'k2_snr', 'stellar_type']
        available_features.extend([f for f in engineered_features if f in df.columns])
        
        # Remove duplicates
        available_features = list(set(available_features))
        
        print(f"📋 Using features: {available_features}")
        
        # Prepare X
        X = df[available_features].copy()
        self.feature_columns = available_features
        
        # Convert object columns to numeric
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare y
        y = df[self.target_column].copy()
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def preprocess_single_sample(self, sample_data):
        """Preprocess a single sample for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([sample_data])
            
            # Handle missing values in the sample
            for col in self.selected_features:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Fill missing values with defaults
            defaults = {
                'pl_orbper': 10.0,
                'pl_rade': 5.0,
                'pl_trandep': 1000.0,
                'pl_trandur': 5.0,
                'pl_insol': 100.0,
                'pl_eqt': 500.0,
                'st_teff': 5000.0,
                'st_rad': 1.0,
                'st_logg': 4.5,
                'sy_vmag': 12.0,
                'sy_dist': 100.0,
                'k2_campaign': 'C1'
            }
            
            for col, default_val in defaults.items():
                if col in df.columns and (df[col].isna().any() or df[col].empty):
                    df[col] = default_val
            
            # Apply same feature engineering
            df = self.k2_feature_engineering(df)
            
            # Select features
            X = df[self.feature_columns].copy() if self.feature_columns else df[self.selected_features].copy()
            
            # Convert object columns
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.factorize(X[col])[0]
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            print(f"❌ Error preprocessing single sample: {e}")
            # Return default scaled features
            default_features = np.zeros((1, len(self.selected_features)))
            return default_features
    
    def create_synthetic_data(self):
        """Create synthetic K2 data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic features
        data = {
            'pl_orbper': np.random.exponential(10, n_samples),
            'pl_rade': np.random.uniform(0.5, 20, n_samples),
            'pl_trandep': np.random.uniform(100, 10000, n_samples),
            'pl_trandur': np.random.uniform(1, 20, n_samples),
            'pl_insol': np.random.uniform(0.1, 1000, n_samples),
            'pl_eqt': np.random.uniform(300, 2000, n_samples),
            'st_teff': np.random.uniform(3000, 7000, n_samples),
            'st_rad': np.random.uniform(0.1, 10, n_samples),
            'st_logg': np.random.uniform(3.5, 5.0, n_samples),
            'sy_vmag': np.random.uniform(9, 16, n_samples),
            'sy_dist': np.random.uniform(10, 1000, n_samples),
            'k2_campaign': [f'C{np.random.randint(0, 20)}' for _ in range(n_samples)],
            'k2_disp': []
        }
        
        # Create synthetic targets based on features
        for i in range(n_samples):
            score = 0
            
            if data['pl_trandep'][i] > 500:
                score += 2
            if 0.5 < data['pl_orbper'][i] < 50:
                score += 1
            if 0.5 < data['pl_rade'][i] < 20:
                score += 1
            
            if score >= 3:
                data['k2_disp'].append('CONFIRMED')
            elif score >= 2:
                data['k2_disp'].append('CANDIDATE')
            else:
                data['k2_disp'].append('FALSE POSITIVE')
        
        df = pd.DataFrame(data)
        
        # Preprocess the synthetic data
        df_clean = self.clean_data(df)
        df_filled = self.handle_missing_values(df_clean)
        df_features = self.k2_feature_engineering(df_filled)
        X, y = self.prepare_features_target(df_features)
        
        print(f"🤖 Created synthetic K2 data: {X.shape}")
        
        return X, y
    
    def save_preprocessor(self, file_path):
        """Save preprocessor configuration"""
        preprocessor_data = {
            'selected_features': self.selected_features,
            'target_column': self.target_column,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        joblib.dump(preprocessor_data, file_path)
        print(f"✅ K2 Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """Load preprocessor configuration"""
        if os.path.exists(file_path):
            preprocessor_data = joblib.load(file_path)
            
            self.selected_features = preprocessor_data.get('selected_features', self.selected_features)
            self.target_column = preprocessor_data.get('target_column', self.target_column)
            self.feature_columns = preprocessor_data.get('feature_columns', [])
            self.scaler = preprocessor_data.get('scaler', StandardScaler())
            
            # Setup label encoder
            classes = preprocessor_data.get('label_encoder_classes', [])
            if classes:
                self.label_encoder.classes_ = np.array(classes)
            
            print(f"✅ K2 Preprocessor loaded from {file_path}")
            return True
        else:
            print(f"⚠️ K2 Preprocessor file not found: {file_path}")
            return False

# Example usage
if __name__ == "__main__":
    # Create a simple K2 model for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("🧪 Testing K2 Preprocessor...")
    
    # Initialize preprocessor
    preprocessor = K2DataPreprocessor()
    
    # Create and preprocess synthetic data
    X, y = preprocessor.create_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training samples: {X_train.shape}")
    print(f"📊 Test samples: {X_test.shape}")
    print(f"🎯 Classes: {preprocessor.label_encoder.classes_}")
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"🎯 Model accuracy: {accuracy:.2f}")
    
    # Save model and preprocessor
    joblib.dump(model, 'k2_model.pkl')
    preprocessor.save_preprocessor('k2_preprocessor.pkl')
    
    print("✅ K2 model and preprocessor saved successfully!")