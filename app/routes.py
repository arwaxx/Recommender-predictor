#routes.py
"""from flask import Flask, request, jsonify
import pickle
from app.model import CraftsmanRecommender

app = Flask(__name__)
craftsman_recommender = None

def load_craftsman_recommender():
    global craftsman_recommender
    if craftsman_recommender is None:
        with open('app/craftsman_recommender.pkl', 'rb') as f:
            craftsman_recommender = pickle.load(f)

def get_recommendations():
    load_craftsman_recommender()

    if request.method == 'GET':
        specialties = request.args.get('specialties', 'default_specialties')
        number_craftsmen = int(request.args.get('number_craftsmen', 3))
    elif request.method == 'POST':
        data = request.get_json()
        specialties = data.get('specialties', 'default_specialties')
        number_craftsmen = int(data.get('number_craftsmen', 3))
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

    recommendations = craftsman_recommender.recommend_by_specialties(specialties, number_craftsmen)

    return jsonify({'recommendations': recommendations})

# Define your routes
app.add_url_rule('/recommendations', 'get_recommendations', get_recommendations, methods=['GET', 'POST'])

if __name__ == "__main__":
    load_craftsman_recommender()
    app.run(debug=True)
"""

"""
# app/routes.py
from flask import Flask, request, jsonify
from app.model import CraftsmanRecommender, load_model
from app.database import fetch_craftsmen_from_database

app = Flask(__name__)
model = None

@app.before_first_request
def load_model_once():
    global model
    if model is None:
        model = load_model('app/craftsman_recommender.pkl')
        model = CraftsmanRecommender(model)  # Assigning the model attribute to CraftsmanRecommender instance

@app.route('/recommendations', methods=['GET', 'POST'])
def get_recommendations():
    global model

    if request.method == 'GET':
        specialties = request.args.get('specialties', 'default_specialties')
        number_craftsmen = int(request.args.get('number_craftsmen', 3))
    elif request.method == 'POST':
        data = request.get_json()
        specialties = data.get('specialties', 'default_specialties')
        number_craftsmen = int(data.get('number_craftsmen', 3))
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

    # Provide database connection details
    host = "localhost"
    user = "root"
    password = ""
    database = "sms"
    port = 3306  # Adjust if using a different port

    # Fetch craftsmen details from the database
    craftsmen_data = fetch_craftsmen_from_database(host, user, password, database, port)

    recommendations = model.recommend_by_specialties(specialties, number_craftsmen, host, user, password, database, port)

    return jsonify({'recommendations': recommendations})
"""

# app/routes.py
from flask import Flask, request, jsonify
from app.model import CraftsmanRecommender, load_model
from app.database import fetch_craftsmen_from_database

app = Flask(__name__)
model = None

@app.before_first_request
def load_model_once():
    global model
    if model is None:
        model = load_model('app/craftsman_recommender.pkl')
        model = CraftsmanRecommender(model)  # Assigning the model attribute to CraftsmanRecommender instance

@app.route('/recommendations', methods=['GET', 'POST'])
def get_recommendations():
    global model

    if request.method == 'GET':
        category = request.args.get('category', 'Carpentry')  # Default category
        number_craftsmen = int(request.args.get('number_craftsmen', 4))  # Default number of craftsmen
    elif request.method == 'POST':
        data = request.get_json()
        category = data.get('category', 'Carpentry')  # Extract category from JSON input
        number_craftsmen = int(data.get('number_craftsmen', 4))  # Extract number of craftsmen
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

    # Provide database connection details
    host = "bisbuk6d0p6wy2iiuazy-mysql.services.clever-cloud.com"
    user = "unzjszxn0e5mkop4"
    password = "Iee3p6UYEQN9b1LS664g"
    database = "bisbuk6d0p6wy2iiuazy"
    port = 3306  # Adjust if using a different port

    recommendations = model.recommend_by_specialties(category, number_craftsmen, host, user, password, database, port)

    return jsonify({'recommendations': recommendations})
