# Ari Butterfield
# asb180007
# Dr. Latifur Khan
# CS 6350.001
# 8 May 2023


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score




# data
data = pd.read_csv('cleanedSpeech.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)



# vectorize text data into tf-idf scores
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train['text'])
test_vectors = vectorizer.transform(test['text'])

# train naive bayes classifier
clf = MultinomialNB()
clf.fit(train_vectors, train['label'])

# evaluate
preds = clf.predict(test_vectors)
accuracy = accuracy_score(test['label'], preds)
print('Accuracy:', accuracy)



