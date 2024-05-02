"""import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class CraftsmanRecommender:
    def __init__(self, data):
        self.data = data
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.craftsmen_matrix, self.craftsman_matrix_train = self._compute_matrices(self.data)

    def _compute_matrices(self, data):
        craftsmen_matrix = self.tfidf_vectorizer.fit_transform(data['TextForRecommendation'])
        craftsman_matrix_train = cosine_similarity(self.tfidf_vectorizer.transform(data['TextForRecommendation']))
        return craftsmen_matrix, craftsman_matrix_train

    def recommend_by_specialties(self, specialties, number_craftsmen):
        similarity_scores = np.zeros(len(self.data))
        recommendations = []

        specialties_vectorized = self.tfidf_vectorizer.transform([specialties])
        
        for i in range(len(self.data)):
            similarity_scores[i] = cosine_similarity(
                specialties_vectorized,
                self.tfidf_vectorizer.transform([self.data['Specialties'].iloc[i]])
            )[0][0]

        similar_indices = similarity_scores.argsort()[:-number_craftsmen-1:-1]

        for x in similar_indices:
            recommendation = {
                "craftsman": self.data['Name'].iloc[x],
                "title": self.data['Title'].iloc[x],
                "rate": self.data['Rate'].iloc[x]
            }
            recommendations.append(recommendation)

        return recommendations

# Load and preprocess data
def preprocess_data(file_path):
    df = pd.read_excel(file_path)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    non_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_columns),
            ('non_numeric', non_numeric_transformer, non_numeric_columns)
        ])

    transformed_data = preprocessor.fit_transform(df)
    column_names = numeric_columns.tolist() + preprocessor.named_transformers_['non_numeric'].get_feature_names_out(non_numeric_columns).tolist()
    dataset = pd.DataFrame(transformed_data, columns=column_names)

    dataset.columns = dataset.columns.str.strip()

    numerical_columns = ['Rate', 'Reviews Number', 'Estimated Price Per $', 'Hired Times']
    dataset_minmax_scaled = dataset.copy()
    minmax_scaler = MinMaxScaler()
    dataset_minmax_scaled[numerical_columns] = minmax_scaler.fit_transform(dataset_minmax_scaled[numerical_columns])

    text_columns = ['Title', 'Profile Description', 'Specialties', 'Payment Methods', 'Reviews']
    dataset_minmax_scaled['TextForRecommendation'] = dataset_minmax_scaled[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

    return dataset_minmax_scaled

# Train and store the model
def train_model(data):
    craftsman_recommender = CraftsmanRecommender(data)

    with open('app/craftsman_recommender.pkl', 'wb') as f:
        pickle.dump(craftsman_recommender, f)

    return craftsman_recommender

if __name__ == "__main__":
    data = preprocess_data('data/craftsman_details2.xlsx')
    train_model(data)
"""
"""
# app/model.py
# app/model.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database import fetch_craftsmen_from_database

class CraftsmanRecommender:
    def __init__(self, model):
        self.model = model

    def recommend_by_specialties(self, specialties, number_craftsmen, host, user, password, database, port):
        # Fetch craftsmen data from the database
        craftsmen_data = fetch_craftsmen_from_database(host, user, password, database, port)

        similarity_scores = np.zeros(len(craftsmen_data))
        recommendations = []

        # Preprocess specialties data
        specialties_vectorized = self.model.tfidf_vectorizer.transform([specialties])

        for i, craftsman in enumerate(craftsmen_data):
            # Preprocess each craftsman's specialties data
            craftsman_specialties_vectorized = self.model.tfidf_vectorizer.transform([craftsman['Specialties']])
            similarity_scores[i] = cosine_similarity(specialties_vectorized, craftsman_specialties_vectorized)[0][0]

        similar_indices = similarity_scores.argsort()[:-number_craftsmen-1:-1]

        for x in similar_indices:
            recommendation = {
                "craftsman": craftsmen_data[x]['Name'],
                "title": craftsmen_data[x]['Title'],
                "rate": craftsmen_data[x]['Rate']
            }
            recommendations.append(recommendation)

        return recommendations

import pickle

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model
"""
"""
# app/model.py
import numpy as np
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database import fetch_craftsmen_from_database

class CraftsmanRecommender:
    def __init__(self, model):
        self.model = model

    def recommend_by_specialties(self, specialties, number_craftsmen, host, user, password, database, port):
        # Fetch craftsmen data from the database
        craftsmen_data = fetch_craftsmen_from_database(host, user, password, database, port)

        similarity_scores = np.zeros(len(craftsmen_data))
        recommendations = []

        # Preprocess specialties data
        specialties_vectorized = self.model.tfidf_vectorizer.transform([specialties])

        # Print transformed input specialties
        print("Transformed Input Specialties:")
        print(specialties_vectorized)

        # Iterate through craftsmen data
        print("Transformed Specialties for Each Craftsman:")
        for i, craftsman in enumerate(craftsmen_data):
            # Preprocess each craftsman's specialties data
            craftsman_specialties_vectorized = self.model.tfidf_vectorizer.transform([craftsman['Specialties']])
            # Compute cosine similarity between specialties
            specialties_similarity = cosine_similarity(specialties_vectorized, craftsman_specialties_vectorized)[0][0]
            # Store the similarity score
            similarity_scores[i] = specialties_similarity
            # Print transformed specialties for debugging
            print(f"Craftsman: {craftsman['Name']}, Transformed Specialties: {craftsman_specialties_vectorized}")

        # Get indices of craftsmen with highest similarity scores
        similar_indices = similarity_scores.argsort()[::-1][:number_craftsmen]

        # Retrieve recommendations based on similarity scores
        for x in similar_indices:
            recommendation = {
                "craftsman": craftsmen_data[x]['Name'],
                "title": craftsmen_data[x]['Title'],
                "rate": craftsmen_data[x]['Rate']
            }
            recommendations.append(recommendation)

        return recommendations

import pickle

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model
"""

import numpy as np
from app.database import fetch_craftsmen_from_database

class CraftsmanRecommender:
    def __init__(self, model):
        self.model = model

    def recommend_by_specialties(self, category, number_craftsmen, host, user, password, database, port):
        # Fetch craftsmen data from the database
        craftsmen_data = fetch_craftsmen_from_database(host, user, password, database, port)

        recommendations = []

        # Filter craftsmen based on the exact category match
        filtered_craftsmen = [craftsman for craftsman in craftsmen_data if craftsman['category'] == category]

        # Limit the number of craftsmen if there are more than requested
        filtered_craftsmen = filtered_craftsmen[:number_craftsmen]

        # Retrieve recommendations
        for craftsman in filtered_craftsmen:
            recommendation = {
                "craftsman": craftsman['f_name'],
                "title": craftsman['l_name'],
                "category": craftsman['category'],
                "address": craftsman['address'],
                "photo": craftsman['photo'],
                "rate": float(craftsman['rate']) if craftsman['rate'] is not None else None,
                "Estimated_Price_Per_Currency": float(craftsman['price']) if craftsman['price'] is not None else None
            }
            recommendations.append(recommendation)

        # Sort recommendations based on rate (handling None values)
        recommendations = sorted(recommendations, key=lambda x: x['rate'] if x['rate'] is not None else float('-inf'), reverse=True)

        return recommendations


import pickle

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

