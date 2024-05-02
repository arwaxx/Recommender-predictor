import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# DataSet
df = pd.read_excel('C:/Users/Arwa/OneDrive/المستندات/crafts2/craftsman_details2.xlsx')

# Data Cleaning
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
dataset['Estimated Price Per $'] = pd.to_numeric(dataset['Estimated Price Per $'], errors='coerce')
dataset['Hired Times'] = pd.to_numeric(dataset['Hired Times'], errors='coerce')

# Data Preprocessing
numerical_columns = ['Rate', 'Reviews Number', 'Estimated Price Per $', 'Hired Times']
dataset_minmax_scaled = dataset.copy()
minmax_scaler = MinMaxScaler()
dataset_minmax_scaled[numerical_columns] = minmax_scaler.fit_transform(dataset_minmax_scaled[numerical_columns])

text_columns = ['Title', 'Profile Description', 'Specialties', 'Payment Methods', 'Reviews']
dataset_minmax_scaled['TextForRecommendation'] = dataset_minmax_scaled[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)

class CraftsmanRecommender:
    def __init__(self, data):
        self.data = data
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.craftsmen_matrix, self.craftsman_matrix_train = self._compute_matrices(self.data)

    def _compute_matrices(self, data):
        craftsmen_matrix = self.tfidf_vectorizer.fit_transform(data['TextForRecommendation'])
        craftsman_matrix_train = cosine_similarity(self.tfidf_vectorizer.transform(data['TextForRecommendation']))
        return craftsmen_matrix, craftsman_matrix_train

    def _print_message(self, craftsman, recom_craftsmen):
        rec_items = len(recom_craftsmen)
        print(f'The {rec_items} recommended craftsmen for {craftsman} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_craftsmen[i]['craftsman']} ({recom_craftsmen[i]['title']}) with Rate: {recom_craftsmen[i]['rate']}")
            print("--------------------")

    def recommend_by_specialties(self, specialties, number_craftsmen):
        similarity_scores = np.zeros(len(self.data))
        recommendations = []

        specialties_vectorized = self.tfidf_vectorizer.transform([specialties])
        print("Input Specialties:", specialties)
        print("Vectorized Specialties:", specialties_vectorized)

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

    def recommend(self, recommendation):
        craftsman = recommendation['craftsman']
        number_craftsmen = recommendation['number_craftsmen']

        # Recompute matrices for each request
        self.craftsmen_matrix, self.craftsman_matrix_train = self._compute_matrices(self.data)

        recom_craftsmen = self.craftsman_matrix_train[craftsman][:number_craftsmen]
        self._print_message(craftsman=craftsman, recom_craftsmen=recom_craftsmen)

# Serialize the CraftsmanRecommender object
with open('craftsman_recommender.pkl', 'wb') as f:
    craftsman_recommender = CraftsmanRecommender(dataset)
    pickle.dump(craftsman_recommender, f)
