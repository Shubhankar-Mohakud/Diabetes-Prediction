"""
Batch Diabetes Risk Prediction
===============================
Predict diabetes risk for multiple patients at once.

Usage: python batch_predict.py
"""

import os
from diabetes_prediction import DiabetesPredictor
import pandas as pd

print("\n" + "="*80)
print(" " * 20 + "BATCH DIABETES RISK PREDICTION")
print("="*80)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'diabetes_best_model.pkl')
data_path = os.path.join(script_dir, 'balanced_diabetes.csv')

# Load the saved model
print("\nLoading trained model...")
predictor = DiabetesPredictor(data_path)
predictor.load_model(model_path)
print("✓ Model loaded successfully!\n")

# ============================================================================
# DEFINE MULTIPLE PATIENTS HERE
# ============================================================================

patients = [
    {
        'name': 'Patient A - High Risk',
        'gender': 'Female',
        'age': 65,
        'hypertension': 1,
        'heart_disease': 1,
        'smoking_history': 'current',
        'bmi': 35.5,
        'HbA1c_level': 7.5,
        'blood_glucose_level': 220
    },
    {
        'name': 'Patient B - Low Risk',
        'gender': 'Male',
        'age': 25,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 22.5,
        'HbA1c_level': 5.2,
        'blood_glucose_level': 95
    },
    {
        'name': 'Patient C - Moderate Risk',
        'gender': 'Female',
        'age': 45,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 28.3,
        'HbA1c_level': 6.0,
        'blood_glucose_level': 140
    },
    {
        'name': 'Patient D',
        'gender': 'Male',
        'age': 55,
        'hypertension': 1,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 30.2,
        'HbA1c_level': 6.5,
        'blood_glucose_level': 160
    },
    {
        'name': 'Patient E',
        'gender': 'Female',
        'age': 38,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 25.8,
        'HbA1c_level': 5.8,
        'blood_glucose_level': 115
    }
]

# ============================================================================
# PROCESS ALL PATIENTS
# ============================================================================

print("="*80)
print(f"Processing {len(patients)} patients...")
print("="*80 + "\n")

results = []
detailed_results = []

for i, patient in enumerate(patients, 1):
    print(f"Analyzing {patient['name']}...")
    
    # Extract name and create prediction dict
    name = patient.pop('name')
    
    # Get prediction
    result = predictor.predict_diabetes_risk(patient)
    
    # Store summary results
    results.append({
        'Patient': name,
        'Age': patient['age'],
        'Gender': patient['gender'],
        'BMI': patient['bmi'],
        'HbA1c': patient['HbA1c_level'],
        'Glucose': patient['blood_glucose_level'],
        'Prediction': 'POSITIVE' if result['prediction_label'] == 1 else 'NEGATIVE',
        'Risk Level': result['risk_level'],
        'Probability': f"{result['diabetes_probability']:.1f}%"
    })
    
    # Store detailed results
    detailed_results.append({
        'name': name,
        'patient': patient,
        'result': result
    })

print("✓ All patients processed!\n")

# ============================================================================
# DISPLAY SUMMARY TABLE
# ============================================================================

print("="*80)
print(" " * 28 + "SUMMARY RESULTS")
print("="*80 + "\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ============================================================================
# DISPLAY DETAILED RESULTS
# ============================================================================

print("\n\n" + "="*80)
print(" " * 26 + "DETAILED RESULTS")
print("="*80 + "\n")

for detail in detailed_results:
    name = detail['name']
    patient = detail['patient']
    result = detail['result']
    
    print("-" * 80)
    print(f"PATIENT: {name}")
    print("-" * 80)
    
    print("\nPatient Information:")
    print(f"  Gender:              {patient['gender']}")
    print(f"  Age:                 {patient['age']} years")
    print(f"  Hypertension:        {'Yes' if patient['hypertension'] else 'No'}")
    print(f"  Heart Disease:       {'Yes' if patient['heart_disease'] else 'No'}")
    print(f"  Smoking History:     {patient['smoking_history']}")
    print(f"  BMI:                 {patient['bmi']}")
    print(f"  HbA1c Level:         {patient['HbA1c_level']}%")
    print(f"  Blood Glucose:       {patient['blood_glucose_level']} mg/dL")
    
    print("\nPrediction Results:")
    print(f"  Diagnosis:           {result['prediction']}")
    print(f"  Risk Level:          {result['risk_level']}")
    print(f"  Diabetes Prob:       {result['diabetes_probability']:.2f}%")
    print(f"  Confidence:          {result['confidence']:.2f}%")
    
    print("\nRecommendation:")
    if result['diabetes_probability'] < 25:
        print("  ✓ LOW RISK - Maintain healthy lifestyle and regular checkups")
    elif result['diabetes_probability'] < 50:
        print("  ⚠ MODERATE RISK - Monitor blood sugar and consult doctor")
    elif result['diabetes_probability'] < 75:
        print("  ⚠⚠ HIGH RISK - Immediate medical consultation recommended")
    else:
        print("  ⚠⚠⚠ VERY HIGH RISK - Urgent medical attention needed")
    
    print()

print("="*80)

# ============================================================================
# SAVE RESULTS TO CSV (OPTIONAL)
# ============================================================================

save_csv = input("\nWould you like to save results to CSV? (Yes/No): ").strip().lower()
if save_csv in ['yes', 'y']:
    filename = input("Enter filename (e.g., results.csv): ").strip()
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    results_df.to_csv(filename, index=False)
    print(f"✓ Results saved to '{filename}'")

print("\n" + "="*80)
print(" " * 28 + "BATCH PREDICTION COMPLETE")
print("="*80 + "\n")

# ============================================================================
# STATISTICS
# ============================================================================

print("STATISTICS:")
print("-"*80)
positive_count = sum(1 for r in results if r['Prediction'] == 'POSITIVE')
negative_count = sum(1 for r in results if r['Prediction'] == 'NEGATIVE')

print(f"Total Patients:        {len(patients)}")
print(f"Positive Predictions:  {positive_count} ({positive_count/len(patients)*100:.1f}%)")
print(f"Negative Predictions:  {negative_count} ({negative_count/len(patients)*100:.1f}%)")

# Risk level breakdown
risk_counts = {}
for r in results:
    risk = r['Risk Level']
    risk_counts[risk] = risk_counts.get(risk, 0) + 1

print("\nRisk Level Breakdown:")
for risk, count in sorted(risk_counts.items()):
    print(f"  {risk:.<20} {count} ({count/len(patients)*100:.1f}%)")

print("="*80 + "\n")
