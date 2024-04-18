from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.satgpacrosses import SATtoGPAModel  # Assuming SATtoGPAModel is a class
from flask_cors import CORS

satgpacross_api = Blueprint('satgpacross_api', __name__, url_prefix='/api/satgpacross')
api = Api(satgpacross_api)
# CORS(satgpacross_api)  # Enable CORS for the API

class SatgpacrossAPI:
    class _CRUD(Resource):
        def post(self):
            print('i think it worked')
            data = request.get_json()
            satscore = data.get('satscore')
            try:
                # Attempt to convert the values to integers
                satscore = int(satscore)
                # gpa = float(gpa)
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid data format"}), 400

            # Instantiate the SATtoGPAModel class with validated values
            model = SATtoGPAModel(satscore=satscore)
            prediction = model.predict()
            print(prediction)
            # Return the prediction as JSON

            return jsonify({"prediction": str(prediction)})

# Add the CRUD resource to the API
api.add_resource(SatgpacrossAPI._CRUD, '/')