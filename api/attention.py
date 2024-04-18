from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

# Assuming AttentionModel is correctly implemented in model.attentions
from model.attentions import AttentionModel

attention_api = Blueprint('attention_api', __name__, url_prefix='/api/attention')
api = Api(attention_api)

class AttentionAPI(Resource):
    def post(self):
        """
        Semantics: In HTTP, POST requests are used to send data to the server for processing.
        Sending passenger data to the server to get a prediction fits the semantics of a POST request.
        
        POST requests send data in the body of the request...
        1. which can handle much larger amounts of data and data types than URL parameters
        2. using an HTTPS request, the data is encrypted, making it more secure
        3. a JSON formatted body is easy to read and write between JavaScript and Python, great for Postman testing
        """
        # Get the passenger data from the request
        score = request.get_json()

        # Get the singleton instance of the AttentionModel
        attentionModel = AttentionModel.get_instance()
        
        # Predict the survival probability of the passenger
        response = attentionModel.predict(score)

        # Return the response as JSON
        return jsonify(response)

api.add_resource(AttentionAPI, '/predict')
