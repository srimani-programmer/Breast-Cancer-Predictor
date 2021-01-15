# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def ann_model_preprocessing():
    #loading the dataset
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
    return X_train,X_test,y_train,y_test

#building te ann model and training it

def ann_model_train(X_train,y_train):
    classifier = Sequential()  # Initialising the ANN

    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))# first layer
    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu')) #first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) #second hidden layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) #output layer
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=1, epochs=100)
    return classifier

#predicting results with the trained model
def ann_model_predict(classifier,X_test):
    y_pred = classifier.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    return y_pred

#accuracy of the model and confusion matrix
def ann_model_acc(y_pred,y_test):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print("Accuracy: " + str(accuracy * 100) + "%")




