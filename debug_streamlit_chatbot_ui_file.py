
import streamlit as st
import regex as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
import joblib
import numpy as np

# Load the Keras model and the TF-IDF vectorizer
model = tf.keras.models.load_model('sentiment_model.keras')
tf_idf_vector = joblib.load('tfidf.pkl')

# Initialize nltk tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to predict sentiment
def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>', '', review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]

    if len(stemmed_review) == 0:
        return 'Input review is empty after preprocessing.'

    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])

    # Debugging information
    print(f"TF-IDF Review Shape: {tfidf_review.shape}")
    print(f"Expected Model Input Shape: {model.input_shape}")
    print(f"TF-IDF Review Type: {type(tfidf_review)}")
    print(f"TF-IDF Review Data: {tfidf_review}")

    if tfidf_review.shape[1] != model.input_shape[1]:
        return f'Input shape mismatch. Expected {model.input_shape[1]}, but got {tfidf_review.shape[1]}.'

    try:
        sentiment_prediction = model.predict(tfidf_review)
    except Exception as e:
        print(f"Model Prediction Error: {str(e)}")
        return f"Prediction error: {str(e)}"

    # Assuming binary classification with classes [0, 1]
    sentiment_label = "Positive" if sentiment_prediction[0][0] > 0.5 else "Negative"

    return sentiment_label

# Streamlit UI
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", sentiment)
