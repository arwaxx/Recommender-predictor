"""from flask import Flask, request, jsonify
import pandas as pd
from haversine import haversine, Unit
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the dataset
data = pd.read_csv(r"D:\Level_4\GP\dataset\pricing\Handyman_Work_Order__HWO__Charges_20240208 (1).csv", encoding='latin1')

# Function to calculate distance from Cairo
def calculate_distance(row):
    cairo_coord = (30.0444, 31.2357)  # Coordinates of Cairo
    location_coord = (row['Latitude'], row['Longitude'])
    distance_km = haversine(cairo_coord, location_coord, unit=Unit.KILOMETERS)
    return distance_km

# Drop irrelevant columns
columns_to_drop = ['ISO', 'Country', 'Locality', 'Street']
data = data.drop(columns=columns_to_drop, axis=1)

# Define numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
non_numeric_columns = data.select_dtypes(exclude=['int64', 'float64']).columns

# Create transformers for numeric and non-numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

non_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

# Create a column transformer to apply different imputation strategies
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_columns),
        ('non_numeric', non_numeric_transformer, non_numeric_columns)
    ])

# Fit and transform the data
dataset = pd.DataFrame(preprocessor.fit_transform(data), columns=numeric_columns.tolist() + preprocessor.named_transformers_['non_numeric'].get_feature_names_out(non_numeric_columns).tolist())

# Encode categorical columns
d = defaultdict(LabelEncoder)
categorical_columns = dataset.select_dtypes(include=['object']).columns
for col in categorical_columns:
    dataset[col] = d[col].fit_transform(dataset[col])

# Split the data into features (X) and target (y)
X = dataset.drop('ChargeAmount', axis=1)
y = dataset['ChargeAmount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Instantiate and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Fit the imputer on the training data
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(X_train)

# Function to make predictions on user input
def predict_charge_amount(latitude, longitude):
    # Create DataFrame with the same columns as the training data
    user_data = pd.DataFrame(columns=X_train.columns)
    # Fill in the user input values
    user_data['Latitude'] = [latitude]
    user_data['Longitude'] = [longitude]
    # Calculate distance from Cairo
    user_data['distance_from_cairo_km'] = user_data.apply(calculate_distance, axis=1)
    
    # Add missing columns in test data that are present in training data
    missing_cols = set(X_train.columns) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0  # Fill missing columns with zeros
    
    # Ensure the order of columns is the same as during training
    user_data = user_data[X_train.columns]
    
    # Preprocess the data to handle missing values
    user_data = imputer.transform(user_data)
    
    prediction = model.predict(user_data)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    prediction = predict_charge_amount(latitude, longitude)
    return jsonify({'predicted_charge_amount': prediction})

"""

# app/predict_model.py
from flask import Blueprint, request, jsonify
import pandas as pd
from haversine import haversine, Unit
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

predict = Blueprint('predict', __name__)

@predict.route('/predict', methods=['POST'])
def predict_handler():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    prediction = predict_charge_amount(latitude, longitude)  # Call the predict_charge_amount function
    return jsonify({'predicted_charge_amount': float(prediction)})  # Convert prediction to float and return as JSON

def predict_charge_amount(latitude, longitude):
    # Load the dataset
    data = pd.read_csv(r"data\Handyman_Work_Order__HWO__Charges_20240208 (1).csv", encoding='latin1')

    # Function to calculate distance from Cairo
    def calculate_distance(row):
        cairo_coord = (30.0444, 31.2357)  # Coordinates of Cairo
        location_coord = (row['Latitude'], row['Longitude'])
        distance_km = haversine(cairo_coord, location_coord, unit=Unit.KILOMETERS)
        return distance_km

    # Drop irrelevant columns
    columns_to_drop = ['ISO', 'Country', 'Locality', 'Street']
    data = data.drop(columns=columns_to_drop, axis=1)

    # Define numeric and non-numeric columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_columns = data.select_dtypes(exclude=['int64', 'float64']).columns

    # Create transformers for numeric and non-numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    non_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    # Create a column transformer to apply different imputation strategies
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_columns),
            ('non_numeric', non_numeric_transformer, non_numeric_columns)
        ])

    # Fit and transform the data
    dataset = pd.DataFrame(preprocessor.fit_transform(data), columns=numeric_columns.tolist() + preprocessor.named_transformers_['non_numeric'].get_feature_names_out(non_numeric_columns).tolist())

    # Encode categorical columns
    d = defaultdict(LabelEncoder)
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dataset[col] = d[col].fit_transform(dataset[col])

    # Split the data into features (X) and target (y)
    X = dataset.drop('ChargeAmount', axis=1)
    y = dataset['ChargeAmount']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

    # Instantiate and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fit the imputer on the training data
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train)

    # Function to make predictions on user input
    def predict_charge_amount_for_coordinates(latitude, longitude):
        # Create DataFrame with the same columns as the training data
        user_data = pd.DataFrame(columns=X_train.columns)
        # Fill in the user input values
        user_data['Latitude'] = [latitude]
        user_data['Longitude'] = [longitude]
        # Calculate distance from Cairo
        user_data['distance_from_cairo_km'] = user_data.apply(calculate_distance, axis=1)
        
        # Add missing columns in test data that are present in training data
        missing_cols = set(X_train.columns) - set(user_data.columns)
        for col in missing_cols:
            user_data[col] = 0  # Fill missing columns with zeros
        
        # Ensure the order of columns is the same as during training
        user_data = user_data[X_train.columns]
        
        # Preprocess the data to handle missing values
        user_data = imputer.transform(user_data)
        
        prediction = model.predict(user_data)
        return prediction[0]  # Return the prediction value, not the function itself

    # Call the function and return the prediction value
    return predict_charge_amount_for_coordinates(latitude, longitude)
