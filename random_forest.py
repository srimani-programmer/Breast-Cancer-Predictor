
# Part 1 - Data Preprocessing

# Importing the libraries
from time import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BreastCancerData.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

t = time()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
output = clf.predict(X_test)
accuracy = accuracy_score(y_test, output)
print("The accuracy of testing data: ", accuracy)
print("The running time: ", time()-t)
