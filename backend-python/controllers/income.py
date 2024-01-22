from models.incomeModel import Income
import json
from bson import ObjectId
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, (ObjectId,)):
        return str(obj)
    elif isinstance(obj, (datetime,)):
        return obj.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    elif isinstance(obj, (dict,)):
        return {key: custom_serializer(value) for key, value in obj.items()}
    else:
        return obj

def addIncome(data):
    title = data.get("title")
    amount = float(data.get("amount"))
    category = data.get("category")
    description = data.get("description")
    date = data.get("date")

    income = Income(
        title=title,
        amount=amount,
        category=category,
        description=description,
        date=date
    )

    try:
        if not all([title, category, description, date]):
            return {"message": "All fields are required!"}, 400

        if amount <= 0 or not isinstance(amount, (int, float)):
            return {"message": "Amount must be a positive number!"}, 400

        income.save()
        return {"message": "Income Added!"}, 200

    except Exception as e:
        return {"message": "Server Error!"}, 500

def getIncomes():
    try:
        incomes = Income.objects().order_by("date")
        incomes_list = list(incomes)

        incomes_list_serialized = json.loads(json.dumps([income.to_mongo().to_dict() for income in incomes_list], default=custom_serializer))
        return incomes_list_serialized
    except Exception as e:
        print(e)
        return {"message": str(e)}, 500

def deleteIncome(id):
    try:
        income = Income.objects.get(id=id)
        income.delete()
        return {"message": "Income deleted!"}, 200
    except Exception as e:
        return {"message": "Server Error!"}, 500
