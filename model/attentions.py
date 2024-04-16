# Import the required libraries for the AttentionModel class
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # unsed for REST API building

# Change variable name and API name and prefix
attention_api = Blueprint('attention_api', __name__,
                   url_prefix='/api/attentions')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(attention_api)

class AttentionModel:
    """A class used to represent the Attention Model for score prediction.
    """
    # a singleton instance of AttentionModel, created to train the model only once, while using it for prediction multiple times
    _instance = None
    
    # constructor, used to initialize the AttentionModel
    def __init__(self):
        # the attention ML model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['subject', 'attention', 'solutions']
        self.target = 'score'
        # load the attention dataset
        self.attention_data = pd.read_csv('attention.csv')

    # clean the attention dataset, prepare it for training
    def _clean(self):
        pass  # No cleaning needed for this dataset

    # train the attention model, using linear regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.attention_data[self.features]
        y = self.attention_data[self.target]
        
        # perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # define the linear regression model
        self.model = LinearRegression()
        
        # train the model
        self.model.fit(X_train, y_train)
        
        # train a decision tree regressor
        self.dt = DecisionTreeRegressor()
        self.dt.fit(X_train, y_train)
        
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionally cleans and builds, the singleton instance of the AttentionModel.
        The model is used for analysis on attention data and predictions on the score of theoretical cases.
        
        Returns:
            AttentionModel: the singleton _instance of the AttentionModel, which contains data and methods for prediction.
        """        
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance

    def predict(self, case):
        """ Predict the score based on the attention case.

        Args:
            case (dict): A dictionary representing an attention case. The dictionary should contain the following keys:
                'subject': The subject ID
                'attention': The type of attention
                'solutions': The number of solutions

        Returns:
           float: predicted score 
        """
        # clean the case data
        case_df = pd.DataFrame(case, index=[0])
        
        # predict the score
        score_prediction = self.model.predict(case_df[self.features])
        # return the score prediction
        return score_prediction[0]
    
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.

        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)} 
    
def initAttention():
    """ Initialize the Attention Model.
    This function is used to load the Attention Model into memory, and prepare it for prediction.
    """
    AttentionModel.get_instance()
    
def testAttention():
    """ Test the Attention Model
    Using the AttentionModel class, we can predict the score based on an attention case.
    Print output of this test contains method documentation, case data, predicted score, and feature weights.
    """
     
    # setup case data for prediction
    print(" Step 1:  Define theoretical case data for prediction: ")
    case = {
        'subject': [6],  
        'attention': ['divided'],
        'solutions': [1]
    }
    print("\t", case)
    print()

    # get an instance of the cleaned and trained Attention Model
    attentionModel = AttentionModel.get_instance()
    print(" Step 2:", attentionModel.get_instance.__doc__)
   
    # print the score prediction
    print(" Step 3:", attentionModel.predict.__doc__)
    score_prediction = attentionModel.predict(case)
    print('\t Predicted Score:', score_prediction)  
    print()
    
    # print the feature weights in the prediction model
    print(" Step 4:", attentionModel.feature_weights.__doc__)
    importances = attentionModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair
        
if __name__ == "__main__":
    print(" Begin:", testAttention.__doc__)
    testAttention()

