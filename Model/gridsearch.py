import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.preprocessing import LabelEncoder
import warnings

from matplotlib import cm as cm

def grid_search_preprocessing():
    data = pd.read_csv('Breast Cancer Data.csv')
    del data['Unnamed: 32']
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values
    # Encoding categorical data
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
#implementing grid search to find best model
def grid_search_train(X_train,y_train):
    models_list = []
    models_list.append(('CART', DecisionTreeClassifier()))
    models_list.append(('SVM', SVC()))
    models_list.append(('NB', GaussianNB()))
    models_list.append(('KNN', KNeighborsClassifier()))
    models_list.append(('Random forest', RandomForestClassifier()))
    num_folds = 10
    results = []
    names = []

    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        start = time.time()
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end - start))
    return results
#visualising grid search results
def grid_search_performance_comp(results):
    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
#after grid search it was found that svc has highest accuracy
def svc_model(X_train,y_train):
    model_svc = SVC(C=2.0, kernel='rbf')
    start = time.time()
    model_svc.fit(X_train, y_train)
    end = time.time()
    print("Run Time: %f" % (end - start))
    return model_svc
#predicting ion svc
def svc_model_predict(model_svc,X_test,y_test):
    predictions = model_svc.predict(X_test)
    print("Accuracy score %f" % accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    




