from models.expenseModel import Expense
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

def addExpense(data):
    title = data.get("title")
    amount = float(data.get("amount"))
    category = data.get("category")
    description = data.get("description")
    date = data.get("date")

    expense = Expense(
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
        
        expense.save()
        return {"message": "Expense Added!"}, 200

    except Exception as e:
        return {"message": "Server Error!"}, 500

def getExpense():
    try:
        
        expenses = Expense.objects().order_by("date")
        expenses_list = list(expenses)
        expense_list_serialized = json.loads(json.dumps([expense.to_mongo().to_dict() for expense in expenses_list], default=custom_serializer))
        return expense_list_serialized
    except Exception as e:
        print(e)
        return {"message": str(e)}, 500    

def deleteExpense(id):
    try:
        expense = Expense.objects.get(id=id)
        expense.delete()
        return {"message": "Expense deleted!"}, 200
    except Exception as e:
        return {"message": "Server Error!"}, 500
