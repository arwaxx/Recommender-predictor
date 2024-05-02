
#predict_charge.py
from flask import Blueprint, request, jsonify
from app.predict_model import predict_charge_amount

predict = Blueprint('predict', __name__)

@predict.route('/predict', methods=['POST'])
def predict_handler():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    prediction = predict_charge_amount(latitude, longitude)  # Call the predict_charge_amount function
    return jsonify({'predicted_charge_amount': float(prediction)})  # Convert prediction to float and return as JSON
