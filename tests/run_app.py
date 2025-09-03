#!/usr/bin/env python3
"""
StreamGuard Fraud Detection Web Application
Run this script to start the Flask web server
"""

import os
import sys
from app import app

def check_models():
    """Check if required model files exist"""
    required_files = [
        'trained_models.pkl',
        'scaler.pkl', 
        'selected_features.pkl',
        'label_encoders.pkl',
        'imputer.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Run 'python fraud_detection_ml_workflow.py' first to train the models")
        print("   The web app will still start but predictions may fail without trained models.\n")
    else:
        print("✅ All required model files found")

def main():
    print("🚀 Starting StreamGuard Fraud Detection Web Application")
    print("=" * 60)
    
    # Check for model files
    check_models()
    
    print(f"🌐 Web application will be available at:")
    print(f"   - Local: http://localhost:5000")
    print(f"   - Network: http://0.0.0.0:5000")
    print(f"   - Demo: http://localhost:5000/demo")
    print("=" * 60)
    print("📊 Features:")
    print("   • Real-time fraud detection")
    print("   • Multiple ML model support")
    print("   • Interactive demo with sample transactions")
    print("   • Modern responsive UI")
    print("=" * 60)
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False  # Disable reloader to avoid duplicate startup messages
        )
    except KeyboardInterrupt:
        print("\n👋 StreamGuard application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
