import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn:")
# Input fields
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
Gender = st.selectbox('Gender', label_encoder.classes_)
Age = st.slider("Age", 18, 90)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])

input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Balance': [Balance],
    'EstimatedSalary': [EstimatedSalary],
    'Tenure': [Tenure],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
})

# Encode Gender
input_data['Gender'] = label_encoder.transform(input_data['Gender'])

# One-hot encode Geography
geo_encoded = one_hot_encoder.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Combine all features
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Reorder columns to match training feature order
feature_order = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
input_data = input_data[feature_order]

# Scale input data using the fitted scaler
input_data_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_prob = prediction[0][0]
    if churn_prob >= 0.5:
        st.error(f"The customer is likely to churn with a probability of {churn_prob:.2f}")
    else:
        st.success(f"The customer is likely to stay with a probability of {1 - churn_prob:.2f}")

    st.write("Model prediction complete.")

