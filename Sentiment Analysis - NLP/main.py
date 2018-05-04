#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:51:52 2018

@author: shubham gupta
"""
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
stopwords = set(w.rstrip() for w in open('stopwords_pruned.txt')) # from http://www.lextek.com/manuals/onix/stopwords1.html and removed some of the stop words
wordnet_lemmatizer = WordNetLemmatizer()

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens
    
clf = linear_model.LogisticRegression(C=88)    
"loading data"
df = pd.read_csv('LabelledData.txt', sep=r',,,',header=None)
df[1] = df[1].map(lambda x: x.strip())

#creating the bag of words model
#encoding target
le = LabelEncoder()
le.fit(df[1])
list(le.classes_) # to check the classes in the dataset
y = le.transform(df[1])

corpus = []
for i in range(0, 1483):
    question = df[0][i]
    question = my_tokenizer(question)
    question = ' '.join(question)
    corpus.append(question)

#using TFIFD 
vectorizer = TfidfVectorizer( norm='l2', min_df=0.001, max_df=0.45, max_features = 5000, ngram_range = ( (1, 2) ), sublinear_tf = True )
vectorizer = vectorizer.fit(corpus)
x = vectorizer.transform(corpus).toarray()
fselect = SelectKBest(chi2 , k=100)
x = fselect.fit_transform(x, y)
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
clf.fit(x_train,y_train)
#applying k-fold cross validation model
accuracies = cross_val_score(estimator = clf, X = x_train, y=y_train, cv = 10)
print("Mean accuracy according to 10-fold cross validation model - %s"%accuracies.mean())
print("Standard deviation according to 10-fold cross validation model - %s"%accuracies.std())

def classify(test_qn):
    test_qn = [test_qn]
    test_list = []
    for qn in test_qn:
        test_list.append(' '.join(my_tokenizer(qn)))
    test_features = vectorizer.transform(test_list).toarray()
    test_features = fselect.transform(test_features)
    pred = clf.predict(test_features)
    return (list(le.inverse_transform(pred)))

# Driver program
if __name__ == '__main__':
    cont = 'y'
    while cont == 'y':
        print ("Enter your Question")
        test_qn = input(str)
        print ("Class : %s"%(classify(test_qn)))
        print ("press y to continue or n to quit")
        cont = input(str)

#Applying grid search to find best model and parameters
"""pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(norm='l2', max_features = 5000, sublinear_tf = True, ngram_range = (1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', linear_model.LogisticRegression()),
])
parameters = {
    'tfidf__min_df':(0.0005, 0.001, 0.0015),
    'tfidf__max_df': (0.40, 0.45, 0.5, 0.65),
    'fselect__k' :[80, 90, 100],
    'clf__C': (82, 85, 86, 87)
}

grid_search_tune = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring = 'accuracy', cv=10, n_jobs=-1)
grid_search_tune.fit(corpus, y)

best_accuracy = grid_search_tune.best_score_
best_param = grid_search_tune.best_params_"""













