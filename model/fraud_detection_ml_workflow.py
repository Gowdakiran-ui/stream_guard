import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
import warnings
import joblib
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')
plt.style.use('default')

class FraudDetectionWorkflow:
    def __init__(self, data_path='train_transaction.csv'):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        self.results = {}
        
    def load_data(self):
        """Load and initial exploration of the dataset"""
        print("=" * 50)
        print("LOADING DATA")
        print("=" * 50)
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset shape: {self.data.shape}")
            print(f"Target distribution:")
            print(self.data['isFraud'].value_counts())
            print(f"Fraud rate: {self.data['isFraud'].mean():.4f}")
            return self.data
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            print("Please ensure the data file exists in the current directory")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def perform_eda(self):
        """Streamlined EDA - skip heavy visualizations"""
        print("\n" + "=" * 50)
        print("EXPLORATORY DATA ANALYSIS (STREAMLINED)")
        print("=" * 50)
        
        # Basic info
        print(f"\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        features_with_missing = (missing_data > 0).sum()
        print(f"\nFeatures with missing values: {features_with_missing}")
        
        if features_with_missing > 0:
            print(f"\nTop 10 features with most missing values:")
            print(missing_df.head(10))
        
        # Data types
        print(f"\nData types distribution:")
        print(self.data.dtypes.value_counts())
        
        # Basic fraud statistics
        fraud_stats = self.data.groupby('isFraud').agg({
            'TransactionAmt': ['count', 'mean', 'median', 'std'],
            'TransactionDT': ['min', 'max']
        }).round(2)
        print(f"\nFraud vs Non-fraud statistics:")
        print(fraud_stats)
        
        # Save EDA summary (skip heavy visualizations)
        eda_summary = {
            'dataset_shape': list(self.data.shape),
            'fraud_rate': float(self.data['isFraud'].mean()),
            'missing_values_summary': missing_df.head(20).to_dict(),
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'basic_stats': fraud_stats.to_dict()
        }
        
        with open('eda_summary.json', 'w') as f:
            json.dump(eda_summary, f, indent=2)
        
        print("\n✅ EDA completed (streamlined). Summary saved to 'eda_summary.json'")
        print("⚡ Skipped heavy visualizations for faster processing")
    
    def preprocess_data(self):
        """Data preprocessing and cleaning"""
        print("\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Make a copy for preprocessing
        df = self.data.copy()
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'TransactionID']
        print(f"Categorical columns: {list(categorical_cols)}")
        
        # Label encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle missing values in categorical columns
            df[col] = df[col].fillna('missing')
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Save label encoders
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['isFraud', 'TransactionID']]
        X = df[feature_cols]
        y = df['isFraud']
        
        # Handle missing values in numerical columns
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Save imputer
        joblib.dump(imputer, 'imputer.pkl')
        
        print(f"Features shape after preprocessing: {X_imputed.shape}")
        print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set fraud rate: {self.y_train.mean():.4f}")
        print(f"Test set fraud rate: {self.y_test.mean():.4f}")
        
        return X_imputed, y
    
    def feature_selection(self):
        """Perform feature selection using multiple methods"""
        print("\n" + "=" * 50)
        print("FEATURE SELECTION")
        print("=" * 50)
        
        # Method 1: Statistical feature selection (SelectKBest)
        print("1. Statistical Feature Selection (SelectKBest)...")
        k_features = min(50, self.X_train.shape[1])  # Ensure k doesn't exceed available features
        selector_stats = SelectKBest(score_func=f_classif, k=k_features)
        X_train_stats = selector_stats.fit_transform(self.X_train, self.y_train)
        selected_features_stats = self.X_train.columns[selector_stats.get_support()].tolist()
        
        # Method 2: Random Forest feature importance
        print("2. Random Forest Feature Importance...")
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_train, self.y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features by importance
        top_k = min(50, len(feature_importance))
        selected_features_rf = feature_importance.head(top_k)['feature'].tolist()
        
        # Method 3: Combine both methods
        selected_features_combined = list(set(selected_features_stats + selected_features_rf))
        
        print(f"Statistical selection: {len(selected_features_stats)} features")
        print(f"Random Forest selection: {len(selected_features_rf)} features")
        print(f"Combined selection: {len(selected_features_combined)} features")
        
        # Use combined features
        self.selected_features = selected_features_combined
        self.feature_importance = feature_importance
        
        # Update training and test sets
        self.X_train = self.X_train[self.selected_features]
        self.X_test = self.X_test[self.selected_features]
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save feature selection results
        feature_selection_results = {
            'selected_features': self.selected_features,
            'feature_importance': feature_importance.head(20).to_dict('records'),
            'selection_methods': ['statistical', 'random_forest', 'combined']
        }
        
        with open('feature_selection_results.json', 'w') as f:
            json.dump(feature_selection_results, f, indent=2)
        
        # Save scaler and selected features
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.selected_features, 'selected_features.pkl')
        
        print("Feature selection completed. Results saved to 'feature_selection_results.json'")
    
    def train_models(self):
        """Train multiple ML models"""
        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                X_train_model = self.X_train_scaled
                X_test_model = self.X_test_scaled
            else:
                X_train_model = self.X_train
                X_test_model = self.X_test
            
            # Train model
            model.fit(X_train_model, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, self.y_train, 
                                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                      scoring='roc_auc', n_jobs=-1)
            
            print(f"{name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model
            self.models[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'use_scaled': name == 'Logistic Regression'
            }
        
        # Save models
        joblib.dump(self.models, 'trained_models.pkl')
        print("\nAll models trained and saved to 'trained_models.pkl'")
    
    def make_predictions(self):
        """Make predictions using trained models"""
        print("\n" + "=" * 50)
        print("MAKING PREDICTIONS")
        print("=" * 50)
        
        predictions = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            use_scaled = model_info['use_scaled']
            
            # Choose appropriate test data
            X_test_model = self.X_test_scaled if use_scaled else self.X_test
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} predictions completed")
        
        self.predictions = predictions
        
        # Save predictions
        pred_df = pd.DataFrame({
            'actual': self.y_test.values
        })
        
        for name in self.models.keys():
            pred_df[f'{name}_pred'] = predictions[name]['y_pred']
            pred_df[f'{name}_proba'] = predictions[name]['y_pred_proba']
        
        pred_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to 'predictions.csv'")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        evaluation_results = {}
        
        for name in self.models.keys():
            y_pred = self.predictions[name]['y_pred']
            y_pred_proba = self.predictions[name]['y_pred_proba']
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'cv_roc_auc_mean': self.models[name]['cv_scores'].mean(),
                'cv_roc_auc_std': self.models[name]['cv_scores'].std()
            }
            
            evaluation_results[name] = metrics
            
            print(f"\n{name} Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']*2:.4f})")
        
        # Save evaluation results
        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create summary comparison
        comparison_df = pd.DataFrame(evaluation_results).T
        comparison_df.to_csv('model_comparison.csv')
        
        print(f"\nModel Comparison Summary:")
        print(comparison_df.round(4))
        print("\nEvaluation results saved to 'evaluation_results.json'")
        print("Model comparison saved to 'model_comparison.csv'")
        
        self.results = evaluation_results
        
        # Find best model
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['roc_auc'])
        print(f"\nBest performing model: {best_model} (ROC-AUC: {evaluation_results[best_model]['roc_auc']:.4f})")
        
        return evaluation_results
    
    def run_complete_workflow(self):
        """Run the complete ML workflow"""
        print("STARTING FRAUD DETECTION ML WORKFLOW (OPTIMIZED)")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Streamlined EDA
            self.perform_eda()
            
            # Step 3: Preprocessing
            self.preprocess_data()
            
            # Step 4: Feature selection
            self.feature_selection()
            
            # Step 5: Train models
            self.train_models()
            
            # Step 6: Make predictions
            self.make_predictions()
            
            # Step 7: Evaluate models
            self.evaluate_models()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total time: {duration}")
            print(f"📁 Output files generated:")
            
            # Create workflow summary
            summary = {
                'completion_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'dataset_shape': list(self.data.shape),
                'models_trained': list(self.models.keys()),
                'best_model': max(self.results.keys(), key=lambda x: self.results[x]['roc_auc']),
                'files_generated': [
                    'eda_summary.json', 'trained_models.pkl', 'predictions.csv',
                    'evaluation_results.json', 'model_comparison.csv',
                    'feature_selection_results.json', 'label_encoders.pkl',
                    'imputer.pkl', 'scaler.pkl', 'selected_features.pkl'
                ]
            }
            
            with open('workflow_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("📋 Workflow summary saved to 'workflow_summary.json'")
            print("\n🚀 Ready to run the Flask web application!")
            print("Run: python run_app.py")
            
        except Exception as e:
            print(f"❌ Error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    # Initialize and run the workflow
    workflow = FraudDetectionWorkflow('train_transaction.csv')
    workflow.run_complete_workflow()
