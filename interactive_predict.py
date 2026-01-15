"""
Interactive Diabetes Risk Prediction
=====================================
This script asks for patient information interactively and provides prediction.

Usage: python interactive_predict.py
"""

import os
from diabetes_prediction import DiabetesPredictor

print("\n" + "="*70)
print(" " * 15 + "DIABETES RISK PREDICTION SYSTEM")
print(" " * 20 + "Interactive Mode")
print("="*70)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'diabetes_best_model.pkl')
data_path = os.path.join(script_dir, 'balanced_diabetes.csv')

# Load the saved model
print("\nLoading trained model...")
predictor = DiabetesPredictor(data_path)
predictor.load_model(model_path)
print("✓ Model loaded successfully!\n")

# Get patient information
print("="*70)
print("Please enter patient information:")
print("="*70)

try:
    gender = input("\nGender (Male/Female): ").strip()
    while gender not in ['Male', 'Female']:
        print("❌ Invalid input. Please enter 'Male' or 'Female'.")
        gender = input("Gender (Male/Female): ").strip()
    
    age = float(input("Age (years): "))
    while age < 0 or age > 120:
        print("❌ Invalid age. Please enter a value between 0 and 120.")
        age = float(input("Age (years): "))
    
    hypertension = input("Hypertension (Yes/No): ").strip().lower()
    while hypertension not in ['yes', 'no', 'y', 'n']:
        print("❌ Invalid input. Please enter 'Yes' or 'No'.")
        hypertension = input("Hypertension (Yes/No): ").strip().lower()
    hypertension = 1 if hypertension in ['yes', 'y'] else 0
    
    heart_disease = input("Heart Disease (Yes/No): ").strip().lower()
    while heart_disease not in ['yes', 'no', 'y', 'n']:
        print("❌ Invalid input. Please enter 'Yes' or 'No'.")
        heart_disease = input("Heart Disease (Yes/No): ").strip().lower()
    heart_disease = 1 if heart_disease in ['yes', 'y'] else 0
    
    print("\nSmoking History options:")
    print("  1. never")
    print("  2. former")
    print("  3. current")
    print("  4. not current")
    print("  5. ever")
    print("  6. No Info")
    smoking_options = ['never', 'former', 'current', 'not current', 'ever', 'No Info']
    smoking_history = input("Enter smoking history: ").strip()
    while smoking_history not in smoking_options:
        print(f"❌ Invalid input. Please enter one of: {', '.join(smoking_options)}")
        smoking_history = input("Enter smoking history: ").strip()
    
    bmi = float(input("BMI (Body Mass Index): "))
    while bmi < 10 or bmi > 60:
        print("❌ Invalid BMI. Please enter a value between 10 and 60.")
        bmi = float(input("BMI (Body Mass Index): "))
    
    hba1c_level = float(input("HbA1c Level (%): "))
    while hba1c_level < 3 or hba1c_level > 15:
        print("❌ Invalid HbA1c level. Please enter a value between 3 and 15.")
        hba1c_level = float(input("HbA1c Level (%): "))
    
    blood_glucose_level = float(input("Blood Glucose Level (mg/dL): "))
    while blood_glucose_level < 50 or blood_glucose_level > 400:
        print("❌ Invalid glucose level. Please enter a value between 50 and 400.")
        blood_glucose_level = float(input("Blood Glucose Level (mg/dL): "))
    
    # Create patient dictionary
    patient = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    
    # Get prediction
    print("\nAnalyzing patient data...")
    result = predictor.predict_diabetes_risk(patient)
    
    # Display results
    print("\n" + "="*70)
    print(" " * 22 + "PREDICTION RESULTS")
    print("="*70)
    
    print("\nPATIENT SUMMARY:")
    print("-"*70)
    print(f"  Gender:                  {patient['gender']}")
    print(f"  Age:                     {patient['age']:.0f} years")
    print(f"  Hypertension:            {'Yes' if patient['hypertension'] else 'No'}")
    print(f"  Heart Disease:           {'Yes' if patient['heart_disease'] else 'No'}")
    print(f"  Smoking History:         {patient['smoking_history']}")
    print(f"  BMI:                     {patient['bmi']:.1f}")
    print(f"  HbA1c Level:             {patient['HbA1c_level']:.1f}%")
    print(f"  Blood Glucose Level:     {patient['blood_glucose_level']:.0f} mg/dL")
    
    print("\n" + "="*70)
    print("DIABETES RISK ASSESSMENT:")
    print("="*70)
    print(f"  Diagnosis:               {result['prediction']}")
    print(f"  Risk Level:              {result['risk_level']}")
    print(f"  Diabetes Probability:    {result['diabetes_probability']:.2f}%")
    print(f"  No Diabetes Probability: {result['no_diabetes_probability']:.2f}%")
    print(f"  Confidence:              {result['confidence']:.2f}%")
    print(f"  Model Used:              {result['model_used']}")
    print("="*70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-"*70)
    if result['diabetes_probability'] < 25:
        print("✓ LOW RISK:")
        print("  • Continue maintaining a healthy lifestyle")
        print("  • Regular exercise and balanced diet")
        print("  • Annual health checkups recommended")
    elif result['diabetes_probability'] < 50:
        print("⚠ MODERATE RISK:")
        print("  • Monitor blood sugar levels regularly")
        print("  • Consult with a healthcare provider")
        print("  • Consider lifestyle modifications")
        print("  • Increase physical activity")
    elif result['diabetes_probability'] < 75:
        print("⚠⚠ HIGH RISK:")
        print("  • Immediate medical consultation recommended")
        print("  • Comprehensive diabetes screening needed")
        print("  • Lifestyle changes are essential")
        print("  • Regular monitoring required")
    else:
        print("⚠⚠⚠ VERY HIGH RISK:")
        print("  • URGENT: Seek immediate medical attention")
        print("  • Comprehensive health evaluation needed")
        print("  • Follow medical advice strictly")
        print("  • Regular monitoring and treatment essential")
    print("="*70 + "\n")
    
    # Ask to predict for another patient
    another = input("\nWould you like to predict for another patient? (Yes/No): ").strip().lower()
    if another in ['yes', 'y']:
        print("\n" * 2)
        import subprocess
        import sys
        subprocess.call([sys.executable, __file__])
    else:
        print("\nThank you for using the Diabetes Risk Prediction System!")
        print("="*70 + "\n")

except KeyboardInterrupt:
    print("\n\n❌ Prediction cancelled by user.")
    print("="*70 + "\n")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("Please check your inputs and try again.")
    print("="*70 + "\n")
