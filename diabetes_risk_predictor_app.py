import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DiabetesRiskPredictor:
    def __init__(self):
        # Load the dataset (assuming a CSV file)
        self.data = pd.read_csv('diabetes_data.csv')
        self.model = RandomForestClassifier()
        self._prepare_data()

    def _prepare_data(self):
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def predict_single(self, input_data):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_array)
        return prediction[0]

    def predict_batch(self, input_data_batch):
        input_array = np.array(input_data_batch)
        predictions = self.model.predict(input_array)
        return predictions.tolist()

if __name__ == '__main__':
    predictor = DiabetesRiskPredictor()
    # Example usage for a single prediction
    single_input = [5, 116, 74, 0, 0, 25.6, 0.201, 30]  # Example input for a single prediction
    print(f'Single Prediction: {predictor.predict_single(single_input)}')

    # Example usage for batch predictions
    batch_input = [[5, 116, 74, 0, 0, 25.6, 0.201, 30], [6, 85, 66, 29, 0, 26.6, 0.351, 31]]  # Example input for batch predictions
    print(f'Batch Predictions: {predictor.predict_batch(batch_input)}')
