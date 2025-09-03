from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

class FraudDetectionApp:
    def __init__(self):
        self.models = None
        self.scaler = None
        self.selected_features = None
        self.label_encoders = None
        self.imputer = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            if os.path.exists('trained_models.pkl'):
                self.models = joblib.load('trained_models.pkl')
                print("Models loaded successfully")
            
            if os.path.exists('scaler.pkl'):
                self.scaler = joblib.load('scaler.pkl')
                print("Scaler loaded successfully")
            
            if os.path.exists('selected_features.pkl'):
                self.selected_features = joblib.load('selected_features.pkl')
                print("Selected features loaded successfully")
            
            if os.path.exists('label_encoders.pkl'):
                self.label_encoders = joblib.load('label_encoders.pkl')
                print("Label encoders loaded successfully")
            
            if os.path.exists('imputer.pkl'):
                self.imputer = joblib.load('imputer.pkl')
                print("Imputer loaded successfully")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def preprocess_input(self, transaction_data):
        """Preprocess input data for prediction"""
        try:
            # Create a DataFrame with the input
            df = pd.DataFrame([transaction_data])
            
            # Handle categorical variables if encoders are available
            if self.label_encoders:
                for col, encoder in self.label_encoders.items():
                    if col in df.columns:
                        # Handle unseen categories
                        try:
                            df[col] = encoder.transform(df[col].fillna('missing'))
                        except ValueError:
                            # If category not seen during training, use the most frequent class
                            df[col] = encoder.transform(['missing'])
            
            # Select only the features used during training
            if self.selected_features:
                # Add missing columns with default values
                for feature in self.selected_features:
                    if feature not in df.columns:
                        df[feature] = 0
                
                df = df[self.selected_features]
            
            # Handle missing values
            if self.imputer:
                df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
            else:
                df_imputed = df.fillna(0)
            
            return df_imputed
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def predict_fraud(self, transaction_data, model_name='Random Forest'):
        """Make fraud prediction"""
        try:
            # Preprocess the input
            processed_data = self.preprocess_input(transaction_data)
            if processed_data is None:
                return None
            
            # Get the model
            if not self.models or model_name not in self.models:
                return None
            
            model_info = self.models[model_name]
            model = model_info['model']
            use_scaled = model_info.get('use_scaled', False)
            
            # Scale data if needed
            if use_scaled and self.scaler:
                processed_data = self.scaler.transform(processed_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            return {
                'prediction': int(prediction),
                'fraud_probability': float(probability[1]),
                'confidence': float(max(probability))
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

# Initialize the fraud detection app
fraud_detector = FraudDetectionApp()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fraud prediction"""
    try:
        data = request.json
        
        # Extract transaction data
        transaction_data = {
            'TransactionAmt': float(data.get('amount', 100)),
            'ProductCD': data.get('product_cd', 'W'),
            'card1': float(data.get('card1', 13926)),
            'card2': float(data.get('card2', 0)),
            'card3': float(data.get('card3', 150)),
            'card4': data.get('card4', 'visa'),
            'card5': float(data.get('card5', 142)),
            'card6': data.get('card6', 'credit'),
            'addr1': float(data.get('addr1', 315)),
            'addr2': float(data.get('addr2', 87)),
            'dist1': float(data.get('dist1', 19)),
            'dist2': float(data.get('dist2', 0)),
            'P_emaildomain': data.get('p_emaildomain', 'gmail.com'),
            'R_emaildomain': data.get('r_emaildomain', ''),
            'C1': float(data.get('c1', 1)),
            'C2': float(data.get('c2', 1)),
            'C3': float(data.get('c3', 0)),
            'C4': float(data.get('c4', 0)),
            'C5': float(data.get('c5', 0)),
            'C6': float(data.get('c6', 1)),
            'C7': float(data.get('c7', 0)),
            'C8': float(data.get('c8', 0)),
            'C9': float(data.get('c9', 1)),
            'C10': float(data.get('c10', 0)),
            'C11': float(data.get('c11', 2)),
            'C12': float(data.get('c12', 0)),
            'C13': float(data.get('c13', 1)),
            'C14': float(data.get('c14', 1)),
            'D1': float(data.get('d1', 14)),
            'D2': float(data.get('d2', 0)),
            'D3': float(data.get('d3', 13)),
            'D4': float(data.get('d4', 0)),
            'D5': float(data.get('d5', 0)),
            'D6': float(data.get('d6', 0)),
            'D7': float(data.get('d7', 0)),
            'D8': float(data.get('d8', 0)),
            'D9': float(data.get('d9', 0)),
            'D10': float(data.get('d10', 13)),
            'D11': float(data.get('d11', 13)),
            'D12': float(data.get('d12', 0)),
            'D13': float(data.get('d13', 0)),
            'D14': float(data.get('d14', 0)),
            'D15': float(data.get('d15', 0))
        }
        
        # Add V features with default values
        for i in range(1, 340):
            transaction_data[f'V{i}'] = float(data.get(f'v{i}', 0))
        
        # Get model name from request
        model_name = data.get('model', 'Random Forest')
        
        # Make prediction
        result = fraud_detector.predict_fraud(transaction_data, model_name)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Format response
        response = {
            'is_fraud': result['prediction'] == 1,
            'fraud_probability': round(result['fraud_probability'] * 100, 2),
            'confidence': round(result['confidence'] * 100, 2),
            'risk_level': 'HIGH' if result['fraud_probability'] > 0.7 else 'MEDIUM' if result['fraud_probability'] > 0.3 else 'LOW',
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with sample transactions"""
    return render_template('demo.html')

@app.route('/api/sample-transactions')
def sample_transactions():
    """API endpoint to get sample transactions for demo"""
    samples = [
        {
            'name': 'Low Risk Transaction',
            'data': {
                'amount': 25.50,
                'product_cd': 'W',
                'card4': 'visa',
                'card6': 'credit',
                'p_emaildomain': 'gmail.com',
                'c1': 1, 'c2': 1, 'c3': 0, 'c4': 0
            }
        },
        {
            'name': 'Medium Risk Transaction',
            'data': {
                'amount': 500.00,
                'product_cd': 'C',
                'card4': 'mastercard',
                'card6': 'debit',
                'p_emaildomain': 'yahoo.com',
                'c1': 2, 'c2': 3, 'c3': 1, 'c4': 1
            }
        },
        {
            'name': 'High Risk Transaction',
            'data': {
                'amount': 2500.00,
                'product_cd': 'R',
                'card4': 'american express',
                'card6': 'credit',
                'p_emaildomain': 'anonymous.com',
                'c1': 5, 'c2': 8, 'c3': 3, 'c4': 2
            }
        }
    ]
    
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
