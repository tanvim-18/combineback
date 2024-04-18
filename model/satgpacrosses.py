import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class SATtoGPAModel:
    def __init__(self, satscore):
        self._satscore = satscore
        # self.target = 'Grade Point Average'

    def predict(self):
        df = pd.read_csv('sattogpa.csv')
        # Ensure the same number of features as the model expects
        X = df['SAT Score']  # Assuming 'SAT Score' is the only feature
        y = df['Grade Point Average']
        regressor = RandomForestRegressor(n_estimators=10, random_state=42)
        regressor.fit(X.values.reshape(-1, 1), y)  # Reshape X to be a 2D array
        # Predicting the GPA
        predicted_gpa = regressor.predict([[self._satscore]])[0]
        return predicted_gpa