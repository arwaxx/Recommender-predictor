"""from flask import Flask
from app.routes import app

if __name__ == '__main__':
    app.run(debug=True)
"""
# run.py

from app.routes import app
from app.model import CraftsmanRecommender, load_model
from app.database import fetch_craftsmen_from_database
from predict_charge import predict  # Update import statement here

app.register_blueprint(predict)

if __name__ == '__main__':
    model = load_model('app/craftsman_recommender.pkl')
    app.run(debug=True)







