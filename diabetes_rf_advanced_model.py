import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle

# Load dataset
dataset_url = 'path_to_diabetes_dataset.csv'  # Update with the actual dataset path
df = pd.read_csv(dataset_url)

# Data Preprocessing
# Update this section with actual preprocessing steps related to your dataset
df.dropna(inplace=True)  # Example of dropping missing values
feature_columns = df.columns[:-1]  # Assuming last column is the target
X = df[feature_columns]
y = df['target_column']  # Update with actual target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]  
}  
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Model
best_rf_model = grid_search.best_estimator_
print(f'Best Hyperparameters: {grid_search.best_params_}')

# Cross-Validation
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')

# Evaluate Model
y_pred = best_rf_model.predict(X_test)
y_prob = best_rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_prob)}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
importance = best_rf_model.feature_importances_
indices = np.argsort(importance)[::-1]  # Sort the feature importances

# Visualize Feature Importance
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importance[indices], align='center')
plt.xticks(range(X.shape[1]), feature_columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Save Model
with open('diabetes_rf_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

# Risk Percentage Prediction
# Based on probability thresholds
risk_percentage = best_rf_model.predict_proba(X_test)[:, 1] * 100  # Convert probability to percentage
y_pred_risk = np.where(risk_percentage > 50, 1, 0)  # Assuming 50% risk as threshold

# Save Predictions
np.savetxt('risk_predictions.csv', risk_percentage, delimiter=',', header='Risk Percentage', comments='')\n