from mongoengine import Document, StringField

class User(Document):
    name = StringField(max_length=50, required=True, trim=True)

    meta = {
        'collection': 'users',
        'timestamps': True,
        'strict': False
        }
