#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[ ]:


app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keyword', methods=['POST','GET'])
def get_response():
    review=request.form['keyword']
    word=[]
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    classifier = joblib.load('classifier.pkl')
    tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
    x=tfidfVectorizer.transform(corpus).toarray()
    pred=classifier.predict(x)
    if (pred[0][0]==0 and pred[0][1]==0 and pred[0][2]==0):
        pred_probs=classifier.predict_proba(x)
        word.append("The likelihood of sentiment is: Negative: {0}%,   Neutral: {1}%,   Positive: {2}%".format(str(int(pred_probs[0][0]*100)), str(int(pred_probs[0][1]*100)), str(int(pred_probs[0][2]*100))))
    if pred[0][0]==1:
        word.append("Sentiment seems to be negative")
    elif pred[0][1]==1:
        word.append("This is neutral")
    elif pred[0][2]==1:
        word.append("This carries a positive sentiment!")

    return render_template('index.html',prediction=word)

if __name__== "__main__":
    app.run(debug=True)





