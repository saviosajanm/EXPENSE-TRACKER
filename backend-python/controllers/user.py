from models.userModel import User
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

def updateName(data):
    name = data.get("name")
    
    try:
        existingData = User.objects.first()

        if not name:
            return {"message": "Cannot leave name as empty!"}, 400

        existingData.name = name
        existingData.save()
        return {"message": "Success!"}, 200

    except Exception as e:
        return {"message": "Server Error!"}, 500

def getName():
    try:
        user = User.objects.first()

        # Serialize user with custom serializer
        user_serialized = json.loads(json.dumps(user.to_mongo().to_dict(), default=custom_serializer))
        return user_serialized
    except Exception as e:
        return {"message": "Server Error!"}, 500

def checkIfName():
    try:
        user = User.objects.first()

        if not user:
            dummyData = User(name='Enter your name')
            dummyData.save()
        
        return {"message": "Success!"}, 200

    except Exception as e:
        return {"message": "Server Error!"}, 500

