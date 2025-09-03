#!/bin/bash
set -e

echo "🐳 StreamGuard Docker Container Starting..."

# Function to check if models exist
check_models() {
    local required_files=("trained_models.pkl" "scaler.pkl" "selected_features.pkl" "label_encoders.pkl" "imputer.pkl")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "/app/$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo "⚠️  Missing model files: ${missing_files[*]}"
        return 1
    else
        echo "✅ All model files found"
        return 0
    fi
}

# Function to train models
train_models() {
    echo "🔄 Training ML models..."
    if [[ -f "/app/train_transaction.csv" ]]; then
        python fraud_detection_ml_workflow.py
        echo "✅ Model training completed"
    else
        echo "❌ Training data not found: train_transaction.csv"
        echo "💡 Please mount the training data file to /app/train_transaction.csv"
        exit 1
    fi
}

# Main logic
if [[ "$1" == "train" ]]; then
    echo "🎯 Running in training mode..."
    train_models
    exit 0
elif [[ "$1" == "app" ]] || [[ "$1" == "" ]]; then
    echo "🌐 Starting web application..."
    
    # Check if models exist, train if needed
    if ! check_models; then
        echo "🔄 Models not found. Starting training..."
        if [[ -f "/app/train_transaction.csv" ]]; then
            train_models
        else
            echo "⚠️  No training data found. Starting app without trained models..."
            echo "💡 Some features may not work until models are trained."
        fi
    fi
    
    # Start the Flask application
    exec python run_app.py
else
    # Execute custom command
    exec "$@"
fi
