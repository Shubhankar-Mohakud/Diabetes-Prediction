"""
Diabetes Risk Prediction System
================================
A streamlined machine learning system for predicting diabetes risk.

Features:
- Multiple ML algorithms comparison
- Model training and evaluation
- Risk prediction for new patients
- Model persistence (save/load)

Usage:
    python diabetes_prediction.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')


class DiabetesPredictor:
    """
    A diabetes prediction system using machine learning.
    
    Attributes:
        data_path (str): Path to the diabetes dataset CSV file
        df (DataFrame): Original dataset
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        scaler (StandardScaler): Feature scaler
        models (dict): Dictionary of trained models
        results (dict): Model performance results
        best_model: Best performing model
    """
    
    def __init__(self, data_path):
        """
        Initialize the DiabetesPredictor.
        
        Args:
            data_path (str): Path to the CSV file containing diabetes data
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_data(self):
        """Load dataset."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Total samples: {self.df.shape[0]:,}")
        print(f"  - Total features: {self.df.shape[1] - 1}")
        print(f"  - Diabetes positive: {self.df['diabetes'].sum():,} ({self.df['diabetes'].mean()*100:.2f}%)")
        print(f"  - Diabetes negative: {(1-self.df['diabetes']).sum():,} ({(1-self.df['diabetes'].mean())*100:.2f}%)")
        
        return self.df
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess data for machine learning.
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)
        
        df_processed = self.df.copy()
        
        # Encode categorical variables
        print("\n1. Encoding categorical variables...")
        
        # Gender encoding
        self.label_encoders['gender'] = LabelEncoder()
        df_processed['gender'] = self.label_encoders['gender'].fit_transform(df_processed['gender'])
        print(f"   ✓ Gender: {dict(enumerate(self.label_encoders['gender'].classes_))}")
        
        # Smoking history encoding
        self.label_encoders['smoking_history'] = LabelEncoder()
        df_processed['smoking_history'] = self.label_encoders['smoking_history'].fit_transform(df_processed['smoking_history'])
        print(f"   ✓ Smoking History: {dict(enumerate(self.label_encoders['smoking_history'].classes_))}")
        
        # Separate features and target
        X = df_processed.drop('diabetes', axis=1)
        y = df_processed['diabetes']
        self.feature_names = X.columns.tolist()
        
        # Split data
        print(f"\n2. Splitting data (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   ✓ Training samples: {self.X_train.shape[0]:,}")
        print(f"   ✓ Testing samples: {self.X_test.shape[0]:,}")
        
        # Scale features
        print("\n3. Scaling features with StandardScaler...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("   ✓ Features scaled successfully")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train Random Forest machine learning model."""
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 80)
        
        # Define model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        name = 'Random Forest'
        print(f"\n{'-' * 80}")
        print(f"Training: {name}")
        print("-" * 80)
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                   cv=5, scoring='accuracy', n_jobs=-1)
        
        # Store results
        self.results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        # Print results
        print(f"✓ Training completed")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\n" + "=" * 80)
        print("✓ MODEL TRAINED SUCCESSFULLY")
        print("=" * 80)
    
    def evaluate_models(self):
        """
        Evaluate the trained Random Forest model.
        
        Returns:
            tuple: (results, model_name)
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Get model results
        model_name = 'Random Forest'
        results = self.results[model_name]
        
        self.best_model_name = model_name
        self.best_model = results['model']
        
        print("\n" + "-" * 80)
        print(f"MODEL PERFORMANCE: {model_name}")
        print("-" * 80)
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  CV Score:  {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        # Detailed classification report
        print(f"\n{'-' * 80}")
        print(f"DETAILED CLASSIFICATION REPORT")
        print("-" * 80)
        y_pred_best = results['y_pred']
        print(classification_report(self.y_test, y_pred_best, 
                                   target_names=['No Diabetes (0)', 'Diabetes (1)']))
        
        # Confusion Matrix
        print(f"\n{'-' * 80}")
        print(f"CONFUSION MATRIX")
        print("-" * 80)
        cm = results['confusion_matrix']
        print(f"                  Predicted")
        print(f"                  No    Yes")
        print(f"Actual   No      {cm[0][0]:<6} {cm[0][1]:<6}")
        print(f"         Yes     {cm[1][0]:<6} {cm[1][1]:<6}")
        
        return results, model_name
    
    def predict_diabetes_risk(self, patient_data):
        """
        Predict diabetes risk for a new patient.
        
        Args:
            patient_data (dict): Patient information with keys:
                - gender: 'Male' or 'Female'
                - age: float (years)
                - hypertension: 0 or 1
                - heart_disease: 0 or 1
                - smoking_history: str ('never', 'former', 'current', etc.)
                - bmi: float
                - HbA1c_level: float (%)
                - blood_glucose_level: float (mg/dL)
        
        Returns:
            dict: Prediction results with probabilities and risk assessment
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Please run train_models() first.")
        
        # Encode categorical variables
        gender_encoded = self.label_encoders['gender'].transform([patient_data['gender']])[0]
        smoking_encoded = self.label_encoders['smoking_history'].transform([patient_data['smoking_history']])[0]
        
        # Prepare input array
        input_data = np.array([[
            gender_encoded,
            patient_data['age'],
            patient_data['hypertension'],
            patient_data['heart_disease'],
            smoking_encoded,
            patient_data['bmi'],
            patient_data['HbA1c_level'],
            patient_data['blood_glucose_level']
        ]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(input_scaled)[0]
        probabilities = self.best_model.predict_proba(input_scaled)[0]
        
        # Determine risk level
        diabetes_prob = probabilities[1] * 100
        risk_level = self._assess_risk_level(diabetes_prob)
        
        result = {
            'prediction': 'POSITIVE - High Risk' if prediction == 1 else 'NEGATIVE - Low Risk',
            'prediction_label': int(prediction),
            'diabetes_probability': diabetes_prob,
            'no_diabetes_probability': probabilities[0] * 100,
            'risk_level': risk_level,
            'model_used': self.best_model_name,
            'confidence': max(probabilities) * 100
        }
        
        return result
    
    def _assess_risk_level(self, probability):
        """
        Assess risk level based on probability.
        
        Args:
            probability (float): Diabetes probability (0-100)
        
        Returns:
            str: Risk level description
        """
        if probability < 25:
            return "LOW RISK ✓"
        elif probability < 50:
            return "MODERATE RISK ⚠"
        elif probability < 75:
            return "HIGH RISK ⚠⚠"
        else:
            return "VERY HIGH RISK ⚠⚠⚠"
    
    def save_model(self, filepath='diabetes_model.pkl'):
        """
        Save the trained model and preprocessing objects.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train models first.")
        
        model_package = {
            'model': self.best_model,
            'model_name': 'Random Forest',
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'results': self.results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        import os
        print(f"\n✓ Model package saved to '{filepath}'")
        print(f"  - Best Model: Random Forest")
        print(f"  - File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    def load_model(self, filepath='diabetes_model.pkl'):
        """
        Load a previously saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.best_model = model_package['model']
        self.best_model_name = model_package['model_name']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.feature_names = model_package['feature_names']
        self.results = model_package['results']
        
        print(f"\n✓ Model loaded from '{filepath}'")
        print(f"  - Model: {self.best_model_name}")
        print(f"  - Ready for predictions!")


def main():
    """
    Main function to run the diabetes prediction pipeline.
    """
    import os
    
    print("\n" + "=" * 80)
    print(" " * 20 + "DIABETES RISK PREDICTION SYSTEM")
    print(" " * 22 + "Machine Learning Application")
    print("=" * 80)
    
    # Update this path to your CSV file location
    DATA_PATH = r"Diabetes Prediction\balanced_diabetes.csv"
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Data file not found at '{DATA_PATH}'")
        print("Please update the DATA_PATH variable with the correct file location.")
        return
    
    # Initialize predictor
    predictor = DiabetesPredictor(DATA_PATH)
    
    # Step 1: Load data
    predictor.load_data()
    
    # Step 2: Preprocess data
    predictor.preprocess_data(test_size=0.2, random_state=42)
    
    # Step 3: Train models
    predictor.train_models()
    
    # Step 4: Evaluate model
    results, best_model_name = predictor.evaluate_models()
    
    # Step 5: Save the best model in the same directory as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'diabetes_best_model.pkl')
    predictor.save_model(model_path)
    
    # Step 6: Example predictions
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS - Testing on New Patients")
    print("=" * 80)
    
    # Example 1: High-risk patient
    patient_1 = {
        'gender': 'Female',
        'age': 65,
        'hypertension': 1,
        'heart_disease': 1,
        'smoking_history': 'current',
        'bmi': 35.5,
        'HbA1c_level': 7.5,
        'blood_glucose_level': 220
    }
    
    print("\n" + "-" * 80)
    print("Patient 1: High-Risk Profile")
    print("-" * 80)
    for key, value in patient_1.items():
        print(f"  {key:.<25} {value}")
    
    result_1 = predictor.predict_diabetes_risk(patient_1)
    print(f"\n🔍 PREDICTION RESULTS:")
    print(f"  Prediction:           {result_1['prediction']}")
    print(f"  Risk Level:           {result_1['risk_level']}")
    print(f"  Diabetes Probability: {result_1['diabetes_probability']:.2f}%")
    print(f"  Model Used:           {result_1['model_used']}")
    print(f"  Confidence:           {result_1['confidence']:.2f}%")
    
    # Example 2: Low-risk patient
    patient_2 = {
        'gender': 'Male',
        'age': 25,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 22.5,
        'HbA1c_level': 5.2,
        'blood_glucose_level': 95
    }
    
    print("\n" + "-" * 80)
    print("Patient 2: Low-Risk Profile")
    print("-" * 80)
    for key, value in patient_2.items():
        print(f"  {key:.<25} {value}")
    
    result_2 = predictor.predict_diabetes_risk(patient_2)
    print(f"\n🔍 PREDICTION RESULTS:")
    print(f"  Prediction:           {result_2['prediction']}")
    print(f"  Risk Level:           {result_2['risk_level']}")
    print(f"  Diabetes Probability: {result_2['diabetes_probability']:.2f}%")
    print(f"  Model Used:           {result_2['model_used']}")
    print(f"  Confidence:           {result_2['confidence']:.2f}%")
    
    # Example 3: Moderate-risk patient
    patient_3 = {
        'gender': 'Female',
        'age': 45,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 28.3,
        'HbA1c_level': 6.0,
        'blood_glucose_level': 140
    }
    
    print("\n" + "-" * 80)
    print("Patient 3: Moderate-Risk Profile")
    print("-" * 80)
    for key, value in patient_3.items():
        print(f"  {key:.<25} {value}")
    
    result_3 = predictor.predict_diabetes_risk(patient_3)
    print(f"\n🔍 PREDICTION RESULTS:")
    print(f"  Prediction:           {result_3['prediction']}")
    print(f"  Risk Level:           {result_3['risk_level']}")
    print(f"  Diabetes Probability: {result_3['diabetes_probability']:.2f}%")
    print(f"  Model Used:           {result_3['model_used']}")
    print(f"  Confidence:           {result_3['confidence']:.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 28 + "ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📊 Generated Files:")
    print("  1. diabetes_best_model.pkl - Trained model for future predictions")
    
    print("\n" + "=" * 80)
    print("\n💡 Usage Tips:")
    print("  - Update DATA_PATH to point to your CSV file")
    print("  - Modify patient_data dictionary to predict for new patients")
    print("  - Use predictor.load_model() to load saved model later")
    print("\n📝 To predict for a new patient:")
    print("  patient = {")
    print("      'gender': 'Female',")
    print("      'age': 50,")
    print("      'hypertension': 1,")
    print("      'heart_disease': 0,")
    print("      'smoking_history': 'former',")
    print("      'bmi': 30.0,")
    print("      'HbA1c_level': 6.5,")
    print("      'blood_glucose_level': 160")
    print("  }")
    print("  result = predictor.predict_diabetes_risk(patient)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import os
    main()