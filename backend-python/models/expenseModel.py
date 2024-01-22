from mongoengine import Document, StringField, FloatField, DateField

class Expense(Document):
    title = StringField(max_length=50, required=True, trim=True)
    amount = FloatField(required=True)
    type = StringField(default="expense")
    date = DateField(required=True, trim=True)
    category = StringField(required=True, trim=True)
    description = StringField(max_length=20, required=True, trim=True)

    meta = {
        'collection': 'expenses',
        'timestamps': True,
        'strict': False
        }
