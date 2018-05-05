
#importing libraries
import pandas as pd

#importing dataset
df = pd.read_csv('Churn_Modelling.csv');
x = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 1] = labelencoder.fit_transform(x[:, 1])
x[:, 2] = labelencoder.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]


#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#importing Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier = Sequential()
classifier.add(Dense(input_dim = 11, init = 'uniform', output_dim = 6, activation = 'relu'))
classifier.add(Dense(init = 'uniform', output_dim = 6, activation = 'relu'))
classifier.add(Dense(init = 'uniform', output_dim = 1, activation = 'sigmoid'))
#Compliling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting ANN to training dataset
classifier.fit(x_train, y_train, batch_size = 10, epochs = 5)


#predicting the test set result
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







'''#fitting via logistic regression
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
print("Standard deviation according to 10-fold cross validation model - %s"%accuracies.std())'''