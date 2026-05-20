import streamlit as st
import joblib
import re
from nltk.corpus import stopwords   

model = joblib.load("src/Fake_News_detection_model.pkl")
vectorizer =joblib.load("src/tfidf_vectorizer.pkl")

stopwords = set(stopwords.words("english"))

def cleaning_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+" , "" , text)
    text = re.sub(r"\d+" , "" ,text)
    text = re.sub(r"[^\w\s]" , "" , text)
    words = text.split()

    cleaned_words = []

    for word in words:
        if word not in stopwords:   
            cleaned_words.append(word)
    return " ".join(cleaned_words)

def predict_news(text):
    cleaned = cleaning_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        return "Real News"
    else:
        return "Fake News"
    

st.title("Fake and Real News Detection")

st.write("Detect whether a news article is Real or Fake using Machine Learning.")


user_input = st.text_area("Enter news test:")


if st.button("Predict"):

    cleaned = cleaning_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    if prediction[0] == 1:
        st.success("Real News")
    else:
        st.error("Fake News")