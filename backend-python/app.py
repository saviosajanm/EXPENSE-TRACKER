from flask import Flask, request, jsonify
from flask_cors import CORS
from db.db import db
import numpy as np
import json

from controllers.expense import addExpense, getExpense, deleteExpense
from controllers.income import addIncome, getIncomes, deleteIncome
from controllers.user import updateName, getName, checkIfName
from model import getPrediction

app = Flask(__name__)
CORS(app)
db()

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(elem) for elem in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

@app.route("/api/v1/add-income", methods=["POST"])
def add_income():
    data = request.get_json()
    return jsonify(addIncome(data))

@app.route("/api/v1/get-incomes", methods=["GET"])
def get_incomes():
    return jsonify(getIncomes())

@app.route("/api/v1/delete-income/<id>", methods=["DELETE"])
def delete_income(id):
    return jsonify(deleteIncome(id))

@app.route("/api/v1/add-expense", methods=["POST"])
def add_expense():
    data = request.get_json()
    return jsonify(addExpense(data))

@app.route("/api/v1/get-expenses", methods=["GET"])
def get_expenses():
    return jsonify(getExpense())

@app.route("/api/v1/delete-expense/<id>", methods=["DELETE"])
def delete_expense(id):
    return jsonify(deleteExpense(id))

@app.route("/api/v1/change-name", methods=["POST"])
def change_name():
    data = request.get_json()
    return jsonify(updateName(data))

@app.route("/api/v1/get-name", methods=["GET"])
def get_name():
    return jsonify(getName())

@app.route("/api/v1/check-name", methods=["GET"])
def check_name():
    return jsonify(checkIfName())

@app.route("/api/v1/prediction", methods=["POST"])
def prediction():
    data = request.get_json()
    choice = data["choice"]
    model = data["model"]
    months = data["months"]
    lookback = data["lookback"]
    ifTrain = data["ifTrain"]
    result, last_month = getPrediction(choice, model, months, lookback, ifTrain)
    #print(type(result), type(last_month), "_________________________________----")
    
    result_json_serializable = convert_to_json_serializable(result)
    last_month_json_serializable = convert_to_json_serializable(last_month)
    #result = convert_to_list(result)
    #last_month = convert_to_list(last_month)
    
    response_data = {
        "prediction": result_json_serializable,
        "last_month": last_month_json_serializable
    }

    # Serialize the result using the custom function
    response = app.response_class(
        response = json.dumps(response_data),
        status=200,
        mimetype="application/json"
    )

    return response

if __name__ == "__main__":
    app.run(debug=True)
