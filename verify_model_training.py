from app.model import preprocess_data, train_model

# Preprocess the data
data = preprocess_data('data/craftsman_details2.xlsx')

# Train the model
craftsman_recommender = train_model(data)

# Verify if the model file is created
import os
model_path = os.path.join(os.getcwd(), 'craftsman_recommender.pkl')
if os.path.exists(model_path):
    print("Model file created successfully!")
else:
    print("Model file not found.")
