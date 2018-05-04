
#importing libraries
import pandas as pd

#importing dataset
df = pd.read_csv('Churn_Modelling.csv');
x = df.iloc[:, [1, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12]].values
y = df.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 2] = labelencoder.fit_transform(x[:, 2])
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [2, 3])
x = onehotencoder.fit_transform(x).toarray()

#splitting the dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#fitting via logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
#find best param
params = {'C':[1e-09, 1e-07, 1e-6, 1e07]}
from sklearn.model_selection import GridSearchCV
grid_search_tune = GridSearchCV(estimator=clf, param_grid=params, scoring = 'accuracy', cv=10, n_jobs=-1)
grid_search_tune.fit(x_train, y_train)

best_accuracy = grid_search_tune.best_score_
best_param = grid_search_tune.best_params_

#using the optimized param - just a test
clf_ = LogisticRegression(C=1e-7)
clf_.fit(x_train, y_train)
# Predicting the Test set results
y_pred = clf_.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


#Model accuracy via k-fold cross validation model - test
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y=y_train, cv = 10)
print("Mean accuracy according to 10-fold cross validation model - %s"%accuracies.mean())
print("Standard deviation according to 10-fold cross validation model - %s"%accuracies.std())