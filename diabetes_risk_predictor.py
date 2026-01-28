import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
data = pd.read_csv('balanced_diabetes_dataset_20260115_223127.csv')

# 2. Preprocess the data
X = data.drop('diabetes', axis=1)  # features
y = data['diabetes']  # target variable

# Define categorical and numerical features
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Create preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 3. Define the model with a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# 6. Prediction function
def predict_diabetes(features):
    """Predict diabetes risk as a percentage."""
    features = pd.DataFrame([features])
    prediction = model.predict_proba(features)[:, 1]
    return prediction * 100  # return percentage

# 7. Save the model
joblib.dump(model, 'diabetes_rf_model.pkl')

# 8. Visualize feature importance
importances = model.named_steps['classifier'].feature_importances_
feature_names = np.concatenate([numerical_features, model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)])
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# 9. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")

# Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
print(f"Best parameters found: {grid_search.best_params_}")
