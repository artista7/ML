This is a machine learning classifier to classify the questions into various categories:-

1. Engineering decisions:-
    Making custom stopwords list
    Lametization instead of stemming
    TFIDF vs Count Vectorizer
    Classifier - Naive Bayes vs KNN vs SVM vs Logistic regression

2. Hyper-Parameter tuning using GridSearch
    TFIDF params
    Logistic Regression params

3. params optimizatoin using gridSearchCV:-
    'tfidf - min_df': 0.001,
    'tfidf - max_df': 0.45,
    'fselect - k' : 100,
    'LogisticRegression classifier - C': 88

4. For accuracy calculation k-fold cross-validation method has been used with CV = 10

4. Result
    Classifier                  Mean Accuracy               Standard Deviation
    1. Gaussian Naive Bayes         82.8%                           3.3%
    2. KNN                          91.8%                           2.2%
    3. Decisions Tree               94.4%                           1.6%
    4. SVM                          95.3%                           1.0%                               
    5. Logistic Regression          96.3%                           1.4%

Basically using custom Stopwords, TFIDF, and logistic regression (along with hyper parameter tuning using grid Search), accuracy of 96% has been achieved