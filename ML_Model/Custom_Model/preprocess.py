import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CustomDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.feature_selector = None
        self.selected_features = None
        self.numeric_features = None
        self.categorical_features = None
        self.target_column = None
        self.feature_columns = None
        self.raw_columns = None
        self.original_target_classes = None
        self.feature_names = None
        self.is_fitted = False
    
    def analyze_dataset(self, df, target_column=None):
        """Analyze the dataset and auto-detect structure"""
        print("🔍 Analyzing dataset structure...")
        
        if df is None or len(df) == 0:
            raise ValueError("Dataset is empty or None")
        
        self.raw_columns = df.columns.tolist()
        
        # Auto-detect target column if not provided
        if target_column is None:
            possible_targets = ['target', 'class', 'label', 'disposition', 'tfopwg_disp', 'koi_disposition', 'species', 'diagnosis', 'outcome']
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    print(f"🎯 Auto-detected target column: {target_column}")
                    break
            
            if target_column is None:
                target_column = df.columns[-1]
                print(f"🎯 Using last column as target: {target_column}")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")
        
        self.target_column = target_column
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col != target_column]
        
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found after removing target column")
        
        self.feature_columns = feature_columns
        
        # Detect feature types
        self.numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df[feature_columns].select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"📊 Dataset Analysis:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Total features: {len(feature_columns)}")
        print(f"   - Numeric features: {len(self.numeric_features)}")
        print(f"   - Categorical features: {len(self.categorical_features)}")
        print(f"   - Target column: {target_column}")
        
        # Handle missing values in target
        target_missing = df[target_column].isna().sum()
        if target_missing > 0:
            print(f"⚠️  Warning: {target_missing} missing values in target column. These will be removed.")
        
        # Target distribution
        target_counts = df[target_column].value_counts()
        print(f"🎯 Target distribution:")
        for value, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {value}: {count} ({percentage:.1f}%)")
        
        self.original_target_classes = target_counts.index.tolist()
        
        return feature_columns
    
    def create_preprocessing_pipeline(self):
        """Create dynamic preprocessing pipeline based on data types"""
        print("🔧 Creating preprocessing pipeline...")
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])
        
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
            print(f"   - Numeric transformer: {len(self.numeric_features)} features")
        
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
            print(f"   - Categorical transformer: {len(self.categorical_features)} features")
        
        if not transformers:
            raise ValueError("No valid features found for preprocessing")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop columns not specified in transformers
            n_jobs=-1
        )
        
        print(f"✅ Preprocessing pipeline created with {len(transformers)} transformers")
    
    def preprocess_data(self, df, target_column=None, test_size=0.2, random_state=42):
        """Complete preprocessing for custom dataset"""
        try:
            feature_columns = self.analyze_dataset(df, target_column)
            
            # Prepare features and target
            X = df[feature_columns]
            y = df[self.target_column]
            
            # Remove samples with missing target
            mask = y.notna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                raise ValueError("No valid samples after removing missing target values")
            
            print(f"📝 After cleaning: {len(X_clean)} samples")
            
            # Encode target variable
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_clean)
            
            print("🎯 Label encoding summary:")
            unique_classes = np.unique(y_encoded)
            for class_idx in unique_classes:
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                count = (y_encoded == class_idx).sum()
                percentage = (count / len(y_encoded)) * 100
                print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
            
            # Create and fit preprocessing pipeline
            self.create_preprocessing_pipeline()
            X_processed = self.preprocessor.fit_transform(X_clean)
            
            # Get feature names
            self.feature_names = self._get_feature_names()
            print(f"📋 Generated {len(self.feature_names)} features after preprocessing")
            
            # Feature selection for high-dimensional data
            if len(self.feature_names) > 15:
                k = min(20, len(self.feature_names) - 5)  # Keep reasonable number of features
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                X_selected = self.feature_selector.fit_transform(X_processed, y_encoded)
                self.selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
                print(f"✅ Selected top {k} features from {len(self.feature_names)} total features")
            else:
                X_selected = X_processed
                self.selected_features = self.feature_names
                print(f"✅ Using all {len(self.feature_names)} features")
            
            # Handle class imbalance
            X_balanced, y_balanced = self.handle_class_imbalance(X_selected, y_encoded)
            
            self.is_fitted = True
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            raise
    
    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        feature_names = []
        
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        if self.categorical_features and hasattr(self.preprocessor, 'named_transformers_'):
            if 'cat' in self.preprocessor.named_transformers_:
                ct = self.preprocessor.named_transformers_['cat']
                if hasattr(ct, 'named_steps') and 'onehot' in ct.named_steps:
                    onehot = ct.named_steps['onehot']
                    cat_features = onehot.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_features)
        
        return feature_names
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using manual resampling"""
        print("⚖️ Handling class imbalance...")
        
        if len(np.unique(y)) < 2:
            print("⚠️  Cannot resample, only one class exists in target column.")
            return X, y
        
        X_df = pd.DataFrame(X)
        X_df['target'] = y
        
        class_counts = X_df['target'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        print("📈 Class distribution before resampling:")
        for class_idx, count in class_counts.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"   {class_name}: {count} samples")
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Resample if significant imbalance
        if imbalance_ratio > 3 and min_count >= 5:  # Only resample if minority class has at least 5 samples
            resampled_dfs = []
            for class_idx in class_counts.index:
                class_df = X_df[X_df['target'] == class_idx]
                n_samples = len(class_df)
                
                if n_samples < max_count:
                    # Upsample minority classes
                    resampled_class = resample(
                        class_df,
                        replace=True,  # Allow sampling with replacement
                        n_samples=max_count,
                        random_state=42
                    )
                    resampled_dfs.append(resampled_class)
                    print(f"   🔼 Upsampled {self.label_encoder.inverse_transform([class_idx])[0]} from {n_samples} to {max_count}")
                else:
                    resampled_dfs.append(class_df)
            
            X_resampled_df = pd.concat(resampled_dfs, ignore_index=True)
            X_resampled = X_resampled_df.drop('target', axis=1).values
            y_resampled = X_resampled_df['target'].values
            
            print("📊 Class distribution after resampling:")
            unique, counts = np.unique(y_resampled, return_counts=True)
            for class_idx, count in zip(unique, counts):
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                print(f"   {class_name}: {count} samples")
                
            return X_resampled, y_resampled
        else:
            if imbalance_ratio > 3:
                print("⚠️  Imbalance detected but minority class too small for resampling")
            else:
                print("📊 Class distribution is balanced, no resampling needed")
            return X, y
    
    def preprocess_single_sample(self, sample_data):
        """Preprocess a single sample for prediction"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Train the model first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized")
        
        # Convert to DataFrame
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        elif isinstance(sample_data, pd.DataFrame):
            sample_df = sample_data.copy()
        else:
            raise ValueError("Sample data must be a dictionary or DataFrame")
        
        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in sample_df.columns:
                # Add missing columns with appropriate default values
                if col in self.numeric_features:
                    sample_df[col] = 0.0  # Default for numeric
                else:
                    sample_df[col] = 'missing'  # Default for categorical
        
        # Reorder columns to match training
        sample_df = sample_df[self.feature_columns]
        
        try:
            # Apply preprocessing
            sample_processed = self.preprocessor.transform(sample_df)
            
            # Apply feature selection if used
            if self.feature_selector:
                sample_processed = self.feature_selector.transform(sample_processed)
            
            return sample_processed
            
        except Exception as e:
            print(f"❌ Error preprocessing sample: {e}")
            raise
    
    def get_preprocessor_summary(self):
        """Get summary information about the preprocessor"""
        if not self.is_fitted:
            return {'error': 'Preprocessor not fitted'}
        
        summary = {
            'target_column': self.target_column,
            'num_original_features': len(self.feature_columns) if self.feature_columns else 0,
            'num_processed_features': len(self.selected_features) if self.selected_features else 0,
            'selected_features': self.selected_features,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'raw_columns': self.raw_columns,
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'is_fitted': self.is_fitted,
            'feature_selector_used': self.feature_selector is not None
        }
        
        return summary
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        joblib.dump(self, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load preprocessor from disk"""
        preprocessor = joblib.load(filepath)
        if not isinstance(preprocessor, cls):
            raise ValueError("Loaded object is not a CustomDataPreprocessor")
        
        print(f"✅ Preprocessor loaded from {filepath}")
        return preprocessor