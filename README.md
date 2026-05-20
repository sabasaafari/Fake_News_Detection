# Fake News Detection

A Machine Learning project for detecting fake news using NLP techniques and TF-IDF.

## Features

- Text preprocessing with NLTK
- TF-IDF vectorization
- Fake vs Real news classification
- Multinomial Naive Bayes model
- Streamlit web app

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Joblib

## Run Project

Install requirements:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run src/app.py
```

## Project Structure

```bash
fake-news-detection/
│
├── data/
├── notebooks/
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Performance
Accuracy: 95%

## Dataset

This project uses a Fake and Real News dataset containing approximately 45,000 news articles.

- Real news articles are mostly collected from Reuters.
- Fake news articles contain misleading or fabricated content.
 
  
## Limitations

The model is trained on a specific dataset structure, so predictions may be biased toward writing style and source patterns. 

## Author
Saba Safari