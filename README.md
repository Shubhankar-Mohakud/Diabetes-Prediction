# Diabetes Risk Prediction System

A machine learning system to predict diabetes risk based on patient health data.

---

## 📋 Files Included

1. **diabetes_prediction.py** - Main module with ML models
2. **predict_patient.py** - Simple prediction script
3. **interactive_predict.py** - Interactive mode with user input
4. **batch_predict.py** - Predict for multiple patients at once
5. **HOW_TO_RUN.py** - Detailed instructions and examples

---

## 🚀 Quick Start

### Step 1: Install Requirements

```bash
pip install pandas numpy scikit-learn
```

### Step 2: Train the Model (First Time Only)

```bash
python diabetes_prediction.py
```

This will:
- Train the Random Forest ML model
- Save the trained model as `diabetes_best_model.pkl`

### Step 3: Make Predictions

Choose one of the following methods:

#### Method 1: Simple Prediction (Recommended)

Edit `predict_patient.py` with your patient data, then run:

```bash
python predict_patient.py
```

#### Method 2: Interactive Mode

```bash
python interactive_predict.py
```

The script will ask for patient information interactively.

#### Method 3: Batch Predictions

Edit `batch_predict.py` with multiple patients, then run:

```bash
python batch_predict.py
```

---

## 📝 Patient Data Format

```python
patient = {
    'gender': 'Female',              # 'Male' or 'Female'
    'age': 55,                       # Years
    'hypertension': 1,               # 0=No, 1=Yes
    'heart_disease': 0,              # 0=No, 1=Yes
    'smoking_history': 'former',     # 'never', 'former', 'current', etc.
    'bmi': 32.5,                     # Body Mass Index
    'HbA1c_level': 6.8,             # Percentage
    'blood_glucose_level': 180       # mg/dL
}
```

---

## 📊 Understanding Results

The prediction provides:

- **Prediction**: POSITIVE (High Risk) or NEGATIVE (Low Risk)
- **Risk Level**: 
  - LOW RISK ✓ (< 25%)
  - MODERATE RISK ⚠ (25-50%)
  - HIGH RISK ⚠⚠ (50-75%)
  - VERY HIGH RISK ⚠⚠⚠ (> 75%)
- **Diabetes Probability**: 0-100%
- **Confidence**: How confident the model is

---

## 💡 Example Usage

### Quick Prediction

```python
from diabetes_prediction import DiabetesPredictor

# Load saved model
predictor = DiabetesPredictor('balanced_diabetes.csv')
predictor.load_model('diabetes_best_model.pkl')

# Define patient
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

# Print result
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['diabetes_probability']:.2f}%")
```

---

## 🎯 Workflow

1. **First Time**: Run `diabetes_prediction.py` to train and save model
2. **Subsequent Use**: Use the saved model (`diabetes_best_model.pkl`) for predictions
3. **No need to retrain** - Just load the model and predict!

---

## ⚙️ Configuration

To change the dataset path, edit the `DATA_PATH` variable in each script:

```python
DATA_PATH = 'balanced_diabetes.csv'  # Change this to your file path
```

---

## 🔍 Troubleshooting

**Error: File not found**
- Make sure `balanced_diabetes.csv` is in the same directory
- Or update the file path in the code

**Error: No module named 'sklearn'**
- Run: `pip install scikit-learn`

**Error: Model not found**
- Train the model first by running: `python diabetes_prediction.py`

---

## 📖 More Information

For detailed instructions, examples, and troubleshooting, see **HOW_TO_RUN.py**

---

## 🏥 Medical Disclaimer

This tool is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## 📞 Support

For questions or issues, refer to the detailed documentation in **HOW_TO_RUN.py**
