from decouple import config
from mongoengine import connect
from pymongo.errors import ConnectionFailure

def db():
    try:
        mongo_url = config('MONGO_URL')
        connect(alias='default', host=mongo_url)
        print("DB Connection Successful")
    except ConnectionFailure as e:
        print("DB Connection Error:", e)
        raise e  # Reraise the exception to be handled in the Flask route

if __name__ == "__main__":
    db()
