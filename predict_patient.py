"""
Simple Diabetes Risk Prediction Script
=======================================
Load a saved model and predict diabetes risk for a patient.

Usage: python predict_patient.py
"""

import os
from diabetes_prediction import DiabetesPredictor

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'diabetes_best_model.pkl')
data_path = os.path.join(script_dir, 'balanced_diabetes.csv')

# Load the saved model
print("Loading trained model...")
predictor = DiabetesPredictor(data_path)
predictor.load_model(model_path)

# ============================================================================
# EDIT THIS SECTION WITH YOUR PATIENT'S DATA
# ============================================================================

patient = {
    'gender': 'Female',                    # 'Male' or 'Female'
    'age': 55,                             # Age in years
    'hypertension': 1,                     # 0 = No, 1 = Yes
    'heart_disease': 0,                    # 0 = No, 1 = Yes
    'smoking_history': 'former',           # 'never', 'former', 'current', 'not current', 'ever', 'No Info'
    'bmi': 32.5,                           # Body Mass Index
    'HbA1c_level': 6.8,                    # HbA1c percentage
    'blood_glucose_level': 180             # Blood glucose in mg/dL
}

# ============================================================================
# GET PREDICTION
# ============================================================================

result = predictor.predict_diabetes_risk(patient)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "="*70)
print(" " * 20 + "DIABETES RISK ASSESSMENT")
print("="*70)

print("\nPATIENT INFORMATION:")
print("-"*70)
print(f"  Gender:                  {patient['gender']}")
print(f"  Age:                     {patient['age']} years")
print(f"  Hypertension:            {'Yes' if patient['hypertension'] else 'No'}")
print(f"  Heart Disease:           {'Yes' if patient['heart_disease'] else 'No'}")
print(f"  Smoking History:         {patient['smoking_history']}")
print(f"  BMI:                     {patient['bmi']}")
print(f"  HbA1c Level:             {patient['HbA1c_level']}%")
print(f"  Blood Glucose Level:     {patient['blood_glucose_level']} mg/dL")

print("\n" + "="*70)
print("PREDICTION RESULTS:")
print("="*70)
print(f"  Diagnosis:               {result['prediction']}")
print(f"  Risk Level:              {result['risk_level']}")
print(f"  Diabetes Probability:    {result['diabetes_probability']:.2f}%")
print(f"  No Diabetes Probability: {result['no_diabetes_probability']:.2f}%")
print(f"  Confidence:              {result['confidence']:.2f}%")
print(f"  Model Used:              {result['model_used']}")
print("="*70 + "\n")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("INTERPRETATION:")
print("-"*70)
if result['diabetes_probability'] < 25:
    print("✓ LOW RISK: The patient has a low probability of diabetes.")
    print("  Recommendation: Maintain healthy lifestyle and regular checkups.")
elif result['diabetes_probability'] < 50:
    print("⚠ MODERATE RISK: The patient shows some risk factors for diabetes.")
    print("  Recommendation: Monitor blood sugar levels and consult a doctor.")
elif result['diabetes_probability'] < 75:
    print("⚠⚠ HIGH RISK: The patient has significant risk factors for diabetes.")
    print("  Recommendation: Immediate medical consultation recommended.")
else:
    print("⚠⚠⚠ VERY HIGH RISK: The patient is at very high risk for diabetes.")
    print("  Recommendation: Urgent medical attention and comprehensive screening needed.")

print("="*70 + "\n")
