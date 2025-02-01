# Ari Butterfield
# asb180007
# Dr. Latifur Khan
# CS 6350.001
# 8 May 2023



# installs used on Google Colab

!pip install pandas
!pip install numpy
!pip install sklearn
!pip install keras
!pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras.callbacks import EarlyStopping

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re



def preprocess_text(text):
    # lowercase and no punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # stem words and lemmatization
    stemmer = SnowballStemmer('english')
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text




train['text'] = train['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)





# data
data = pd.read_csv('cleanedSpeech.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)




# tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])
train_sequences = tokenizer.texts_to_matrix(train['text'], mode='binary')
test_sequences = tokenizer.texts_to_matrix(test['text'], mode='binary')

# labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train['label'])
test_labels = label_encoder.transform(test['label'])

# model architecture
model = Sequential()
model.add(Dense(units=256, input_dim=len(tokenizer.word_index)+1, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# complie model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_sequences, train_labels, validation_split=0.2, epochs=10, batch_size=64, callbacks=[early_stop])

# evaluate
loss, accuracy = model.evaluate(test_sequences, test_labels)
print('Accuracy:', accuracy)
