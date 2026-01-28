# diabetes_model_complete.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the dataset
def preprocess_data(data):
    # Handling missing values (if any)
    data.fillna(data.median(), inplace=True)
    
    # Add more preprocessing steps as needed
    return data

# Train Random Forest model
def train_model(X_train, y_train):
    rf = RandomForestClassifier()
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, predictions))
    print("ROC AUC Score:", roc_auc_score(y_test, predictions_proba))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return predictions_proba

# Visualize feature importance
def visualize_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

# Main function
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('path_to_your_data.csv')
    data = preprocess_data(data)
    
    # Define features and target variable
    X = data.drop('target', axis=1)  # replace 'target' with your target column
    y = data['target']  # replace 'target' with your target column
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = train_model(X_train, y_train)

    # Evaluate the model
    preds_proba = evaluate_model(model, X_test, y_test)
    
    # Visualize feature importance
    visualize_importance(model, X.columns)
    
    # Save the model
    joblib.dump(model, 'diabetes_random_forest_model.pkl')