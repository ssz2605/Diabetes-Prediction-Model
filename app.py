"""
Diabetes Prediction Flask Web Application
Author: Shreya Saxena
GitHub: https://github.com/ssz2605/Diabetes-Prediction-Model
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Configure app
app.config['JSON_SORT_KEYS'] = False

# Define paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'models' / 'best_diabetes_model.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'scaler.pkl'

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model files not found. Please train the model first.")
    model = None
    scaler = None


# Route: Home page
@app.route('/')
def home():
    """Render home page with prediction form"""
    return render_template('index.html')


# Route: Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Expected JSON format:
    {
        "pregnancies": float,
        "glucose": float,
        "blood_pressure": float,
        "skin_thickness": float,
        "insulin": float,
        "bmi": float,
        "diabetes_pedigree_function": float,
        "age": float
    }
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that all fields are present
        required_fields = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Extract and validate input values
        try:
            pregnancies = float(data['pregnancies'])
            glucose = float(data['glucose'])
            blood_pressure = float(data['blood_pressure'])
            skin_thickness = float(data['skin_thickness'])
            insulin = float(data['insulin'])
            bmi = float(data['bmi'])
            diabetes_pedigree = float(data['diabetes_pedigree_function'])
            age = float(data['age'])
        except ValueError:
            return jsonify({'error': 'All inputs must be numeric'}), 400
        
        # Validate ranges (medical constraints)
        if not (0 <= pregnancies <= 20):
            return jsonify({'error': 'Pregnancies should be between 0 and 20'}), 400
        if not (0 <= glucose <= 200):
            return jsonify({'error': 'Glucose should be between 0 and 200'}), 400
        if not (0 <= blood_pressure <= 150):
            return jsonify({'error': 'Blood Pressure should be between 0 and 150'}), 400
        if not (0 <= skin_thickness <= 100):
            return jsonify({'error': 'Skin Thickness should be between 0 and 100'}), 400
        if not (0 <= insulin <= 900):
            return jsonify({'error': 'Insulin should be between 0 and 900'}), 400
        if not (10 <= bmi <= 70):
            return jsonify({'error': 'BMI should be between 10 and 70'}), 400
        if not (0 <= diabetes_pedigree <= 2.5):
            return jsonify({'error': 'Diabetes Pedigree should be between 0 and 2.5'}), 400
        if not (1 <= age <= 120):
            return jsonify({'error': 'Age should be between 1 and 120'}), 400
        
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({'error': 'Model not available. Please train first.'}), 500
        
        # Prepare feature array (same order as training)
        features = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'confidence': float(probability[prediction] * 100),
            'positive_probability': float(probability[1] * 100),
            'negative_probability': float(probability[0] * 100)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# Route: Health check
@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({'status': 'OK', 'model_loaded': model is not None}), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print(" Starting Diabetes Prediction App...")
    print("📍 Running on http://localhost:5000")
    print("💡 Make sure you have trained the model first!")
    app.run(debug=True, host='localhost', port=5000)
