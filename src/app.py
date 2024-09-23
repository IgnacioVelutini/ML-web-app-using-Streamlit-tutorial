from utils import db_connect
engine = db_connect()

# your code here

import streamlit as st
import pickle

# Load the saved model and vectorizer from ../models/
model = pickle.load(open('../models/logistic_regression_model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

st.title('Sentiment Analysis App')

# Input field for the user to enter text
user_input = st.text_area("Enter a review")

if st.button('Predict'):
    if user_input:
        # Transform the input text using the vectorizer
        transformed_input = vectorizer.transform([user_input])
        # Predict the sentiment
        prediction = model.predict(transformed_input)
        st.write(f'The predicted sentiment is: {prediction[0]}')
    else:
        st.write("Please enter a review to analyze.")
