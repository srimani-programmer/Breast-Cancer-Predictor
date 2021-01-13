import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##selecting x and y
dataset = pd.read_csv('Breast Cancer Data.csv')
X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values

#Dimension
print("Cancer data set dimensions : {}".format(dataset.shape))


##Null or na values
dataset.isnull().sum()
dataset.isna().sum()

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

##Predicting with the test dataset

Y_pred = classifier.predict(X_test)
print() 
print("Total test observations :",Y_pred.shape[0] )

print()

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print()
##Accuracy
print("Accuracy :",(cm[0][0]+cm[1][1])/Y_pred.shape[0])
print()