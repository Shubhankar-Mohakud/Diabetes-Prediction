# Diabetes Prediction Random Forest Model

## Overview
This repository contains a Random Forest model for predicting diabetes based on various health metrics. The model aims to assist healthcare professionals by providing insights into the likelihood of diabetes in patients.

## Features
- **Input Features:**
  - Age
  - Body Mass Index (BMI)
  - Blood Pressure
  - Insulin Level
  - Glucose Level
  - Skin Thickness
  - Diabetes Pedigree Function
  - Physical Activity Level

- **Model Type:** Random Forest Regressor

## Model Training
1. **Data Preprocessing:**
   - The dataset was cleaned and preprocessed, including handling missing values and encoding categorical variables.
   - Features were scaled to ensure uniformity and improve model performance.

2. **Train-Test Split:**
   - The dataset was split into training and testing sets, with 70% for training and 30% for testing.

3. **Training the Model:**
   - The Random Forest model was trained using the training set. Hyperparameters such as number of trees and depth were optimized using Grid Search.

4. **Cross-Validation:**
   - K-Fold Cross-Validation was employed to ensure model reliability and accuracy.

## Usage
To use the trained model for predictions, follow these steps:
1. Load the model using the appropriate libraries (e.g., `joblib` for Python).
2. Provide the required features as input.
3. Utilize the `predict` method to get the diabetes prediction.

Example code to load the model and make predictions:
```python
import joblib

# Load the model
model = joblib.load('diabetes_model.pkl')

# Input features
input_features = [[age, bmi, blood_pressure, insulin, glucose, skin_thickness, pedigree, physical_activity]]

# Make a prediction
prediction = model.predict(input_features)
```

## Evaluation Metrics
The performance of the Random Forest model was evaluated using the following metrics:
- **Accuracy:** Percentage of correct predictions.
- **Precision:** Measure of the model's exactness.
- **Recall:** Measure of the model's completeness.
- **F1 Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve to evaluate the model's performance across different thresholds.

## Conclusion
The Random Forest model shows promising results in predicting diabetes based on health metrics. Continuous monitoring and updating of the model with new data are recommended to improve accuracy and performance.