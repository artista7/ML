#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:26:01 2018

@author: bluedanube
"""
# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split

df = pd.read_csv('LabelledData.txt', sep=r',,,',header=None)
df[1] = df[1].map(lambda x: x.strip())
#df[1].value_counts()

"cleaning data"
# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords_pruned.txt'))
wordnet_lemmatizer = WordNetLemmatizer()

#from nltk.stem.porter import PorterStemmer
#ps = PorterStemmer()
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    #tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    #tokens = [ps.stem(word) for word in tokens] 
    return tokens

corpus = []
for i in range(0, 1483):
    question = df[0][i]
    question = my_tokenizer(question)
    question = ' '.join(question)
    corpus.append(question)
    
#creating the bag of words model
le = LabelEncoder()
le.fit(df[1])
list(le.classes_) # to check the classes in the dataset
y = le.transform(df[1])

#Applying optimized params in TF-IFD
vectorizer = TfidfVectorizer( norm='l2', min_df=0.001, max_df=0.45, max_features = 5000, ngram_range = ( (1, 2) ), sublinear_tf = True )
vectorizer = vectorizer.fit(corpus)
x = vectorizer.transform(corpus).toarray()
fselect = SelectKBest(chi2 , k=500)
x = fselect.fit_transform(x, y)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# truncate and pad input sequences
max_review_length = 500
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





