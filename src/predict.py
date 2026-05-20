import joblib
import re
from nltk.corpus import stopwords   

model = joblib.load("./src/Fake_News_detection_model.pkl")
vectorizer =joblib.load("./src/tfidf_vectorizer.pkl")

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
    

sample_news = "This is a simple news article about economy growth"
print(predict_news(sample_news))
