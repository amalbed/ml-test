from flask import request
from flask_restx import Api, Resource, fields
from .model_manager import molecule


rest_api = Api(version="1.0", title="Smiles API")


"""
    Flask-Restx models for api request and response data
"""

molecule_model = rest_api.model(
    "MoleculeModel",
    {
        "id": fields.String(required=False, min_length=2, max_length=64),
        "smile": fields.String(required=True, min_length=2, max_length=64),
    },
)


"""
    Flask-Restx routes
"""


@rest_api.route("/api/predict", methods=["POST"])
class Predict(Resource):
    """
    Predicts P1 feature for a given smile
    """

    @rest_api.expect(molecule_model, validate=True)
    def post(self):
        req_data = request.get_json()

        _smile = req_data.get("smile")

        return {
            "success": True,
            "feature": molecule.predict(self, _smile),
            "msg": "The feature was successfully predicted",
        }, 200
