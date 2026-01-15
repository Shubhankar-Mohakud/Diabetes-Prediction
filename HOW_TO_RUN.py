"""
================================================================================
HOW TO RUN THE DIABETES PREDICTION SYSTEM
================================================================================

This guide explains how to train models and predict diabetes risk for patients.

================================================================================
STEP 1: SETUP & INSTALLATION
================================================================================

1. Install required libraries:
   
   pip install pandas numpy scikit-learn

   OR if you have a requirements.txt:
   
   pip install -r requirements.txt

2. Make sure you have:
   - diabetes_prediction.py
   - balanced_diabetes.csv (your dataset)

================================================================================
STEP 2: FIRST TIME - TRAIN THE MODEL
================================================================================

Option A: Run the complete program (recommended for first time)
---------------------------------------------------------------

Simply run:

    python diabetes_prediction.py

This will:
- Load the dataset
- Train 5 different ML models
- Compare their performance
- Save the best model as 'diabetes_best_model.pkl'
- Show example predictions

Option B: Train programmatically
---------------------------------

Create a new Python file (e.g., train_model.py):

    from diabetes_prediction import DiabetesPredictor
    
    # Initialize predictor
    predictor = DiabetesPredictor('balanced_diabetes.csv')
    
    # Load and preprocess data
    predictor.load_data()
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate and select best model
    predictor.evaluate_models()
    
    # Save the model
    predictor.save_model('my_diabetes_model.pkl')

Then run:

    python train_model.py

================================================================================
STEP 3: MAKE PREDICTIONS FOR NEW PATIENTS
================================================================================

METHOD 1: Using the trained predictor (in the same session)
-----------------------------------------------------------

After training, you can immediately predict:

    # Patient data
    patient = {
        'gender': 'Female',
        'age': 55,
        'hypertension': 1,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 32.5,
        'HbA1c_level': 6.8,
        'blood_glucose_level': 180
    }
    
    # Get prediction
    result = predictor.predict_diabetes_risk(patient)
    
    # Display results
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Diabetes Probability: {result['diabetes_probability']:.2f}%")


METHOD 2: Load saved model and predict (new session - RECOMMENDED)
------------------------------------------------------------------

Create a new file (e.g., predict_patient.py):

    import os
    from diabetes_prediction import DiabetesPredictor
    
    # Get the directory where the model is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'diabetes_best_model.pkl')
    data_path = os.path.join(script_dir, 'balanced_diabetes.csv')
    
    # Create predictor instance
    predictor = DiabetesPredictor(data_path)
    
    # Load the saved model (no training needed!)
    predictor.load_model(model_path)
    
    # Define patient data
    patient = {
        'gender': 'Male',
        'age': 45,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 27.3,
        'HbA1c_level': 5.9,
        'blood_glucose_level': 120
    }
    
    # Get prediction
    result = predictor.predict_diabetes_risk(patient)
    
    # Print results
    print("\n" + "="*60)
    print("DIABETES RISK PREDICTION")
    print("="*60)
    print(f"\nPatient Information:")
    for key, value in patient.items():
        print(f"  {key:.<25} {value}")
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Prediction:           {result['prediction']}")
    print(f"Risk Level:           {result['risk_level']}")
    print(f"Diabetes Probability: {result['diabetes_probability']:.2f}%")
    print(f"Confidence:           {result['confidence']:.2f}%")
    print(f"Model Used:           {result['model_used']}")
    print("="*60 + "\n")

Then run:

    python predict_patient.py


METHOD 3: Interactive prediction script
---------------------------------------

Create a file (e.g., interactive_predict.py):

    from diabetes_prediction import DiabetesPredictor
    
    # Load model
    predictor = DiabetesPredictor('balanced_diabetes.csv')
    predictor.load_model('diabetes_best_model.pkl')
    
    print("\n" + "="*60)
    print("DIABETES RISK PREDICTION - INTERACTIVE MODE")
    print("="*60)
    
    # Get input from user
    print("\nEnter patient information:")
    gender = input("Gender (Male/Female): ")
    age = float(input("Age (years): "))
    hypertension = int(input("Hypertension (0=No, 1=Yes): "))
    heart_disease = int(input("Heart Disease (0=No, 1=Yes): "))
    smoking_history = input("Smoking History (never/former/current/not current/ever/No Info): ")
    bmi = float(input("BMI: "))
    hba1c_level = float(input("HbA1c Level (%): "))
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
    result = predictor.predict_diabetes_risk(patient)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPrediction:           {result['prediction']}")
    print(f"Risk Level:           {result['risk_level']}")
    print(f"Diabetes Probability: {result['diabetes_probability']:.2f}%")
    print(f"Confidence:           {result['confidence']:.2f}%")
    print(f"Model Used:           {result['model_used']}")
    print("="*60 + "\n")

Then run:

    python interactive_predict.py


METHOD 4: Batch predictions for multiple patients
-------------------------------------------------

Create a file (e.g., batch_predict.py):

    from diabetes_prediction import DiabetesPredictor
    import pandas as pd
    
    # Load model
    predictor = DiabetesPredictor('balanced_diabetes.csv')
    predictor.load_model('diabetes_best_model.pkl')
    
    # Define multiple patients
    patients = [
        {
            'name': 'Patient A',
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
            'name': 'Patient B',
            'gender': 'Male',
            'age': 30,
            'hypertension': 0,
            'heart_disease': 0,
            'smoking_history': 'never',
            'bmi': 23.5,
            'HbA1c_level': 5.3,
            'blood_glucose_level': 100
        },
        {
            'name': 'Patient C',
            'gender': 'Female',
            'age': 50,
            'hypertension': 0,
            'heart_disease': 0,
            'smoking_history': 'former',
            'bmi': 29.0,
            'HbA1c_level': 6.2,
            'blood_glucose_level': 145
        }
    ]
    
    # Predict for all patients
    results = []
    for patient in patients:
        name = patient.pop('name')  # Remove name before prediction
        result = predictor.predict_diabetes_risk(patient)
        results.append({
            'Name': name,
            'Age': patient['age'],
            'BMI': patient['bmi'],
            'Prediction': result['prediction'],
            'Risk Level': result['risk_level'],
            'Diabetes Probability': f"{result['diabetes_probability']:.2f}%"
        })
    
    # Display results as a table
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BATCH PREDICTION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

Then run:

    python batch_predict.py

================================================================================
STEP 4: UNDERSTANDING THE INPUT PARAMETERS
================================================================================

When creating patient data, use these formats:

patient = {
    'gender': str,              # 'Male' or 'Female'
    'age': float,               # Age in years (e.g., 45.0)
    'hypertension': int,        # 0 (No) or 1 (Yes)
    'heart_disease': int,       # 0 (No) or 1 (Yes)
    'smoking_history': str,     # 'never', 'former', 'current', 
                                # 'not current', 'ever', 'No Info'
    'bmi': float,               # Body Mass Index (e.g., 27.5)
    'HbA1c_level': float,       # HbA1c percentage (e.g., 6.5)
    'blood_glucose_level': float # mg/dL (e.g., 140.0)
}

================================================================================
STEP 5: UNDERSTANDING THE OUTPUT
================================================================================

The prediction returns a dictionary with:

{
    'prediction': str,               # 'POSITIVE - High Risk' or 
                                     # 'NEGATIVE - Low Risk'
    
    'prediction_label': int,         # 0 (No Diabetes) or 1 (Diabetes)
    
    'diabetes_probability': float,   # 0-100 (probability of diabetes)
    
    'no_diabetes_probability': float, # 0-100 (probability of no diabetes)
    
    'risk_level': str,               # 'LOW RISK ✓'
                                     # 'MODERATE RISK ⚠'
                                     # 'HIGH RISK ⚠⚠'
                                     # 'VERY HIGH RISK ⚠⚠⚠'
    
    'model_used': str,               # Name of the ML model used
    
    'confidence': float              # 0-100 (confidence in prediction)
}

Risk Level Classification:
- LOW RISK:       Probability < 25%
- MODERATE RISK:  Probability 25-50%
- HIGH RISK:      Probability 50-75%
- VERY HIGH RISK: Probability > 75%

================================================================================
QUICK START EXAMPLES
================================================================================

Example 1: Complete workflow (train + predict)
----------------------------------------------

    from diabetes_prediction import DiabetesPredictor
    
    # Train
    predictor = DiabetesPredictor('balanced_diabetes.csv')
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.save_model('my_model.pkl')
    
    # Predict
    patient = {
        'gender': 'Female', 'age': 55, 'hypertension': 1,
        'heart_disease': 0, 'smoking_history': 'former',
        'bmi': 32.5, 'HbA1c_level': 6.8, 'blood_glucose_level': 180
    }
    result = predictor.predict_diabetes_risk(patient)
    print(f"Risk: {result['risk_level']} ({result['diabetes_probability']:.1f}%)")


Example 2: Quick prediction (using saved model)
-----------------------------------------------

    from diabetes_prediction import DiabetesPredictor
    
    # Load saved model
    predictor = DiabetesPredictor('balanced_diabetes.csv')
    predictor.load_model('diabetes_best_model.pkl')
    
    # Predict
    patient = {'gender': 'Male', 'age': 30, 'hypertension': 0,
               'heart_disease': 0, 'smoking_history': 'never',
               'bmi': 23.5, 'HbA1c_level': 5.3, 
               'blood_glucose_level': 100}
    
    result = predictor.predict_diabetes_risk(patient)
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")

================================================================================
TROUBLESHOOTING
================================================================================

Error: "File not found"
-----------------------
Solution: Make sure the CSV file path is correct in the code.

Error: "No module named 'sklearn'"
----------------------------------
Solution: Install scikit-learn: pip install scikit-learn

Error: "KeyError" when predicting
---------------------------------
Solution: Make sure all required fields are in the patient dictionary.

Error: Value not in encoder classes
------------------------------------
Solution: Check that gender is 'Male' or 'Female' and smoking_history 
matches one of: 'never', 'former', 'current', 'not current', 'ever', 'No Info'

================================================================================
TIPS & BEST PRACTICES
================================================================================

1. Train the model ONCE and save it. Then load it for predictions.
2. Use the saved model for all future predictions (faster).
3. Keep the .pkl file safe - it contains your trained model.
4. Validate input data before prediction to avoid errors.
5. Check the confidence score - higher confidence = more reliable prediction.

================================================================================
"""

# Save this as a reference guide
if __name__ == "__main__":
    print(__doc__)
