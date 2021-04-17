# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Breast Cancer Data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#TRAINING THE MODEL USING SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
svc_model= SVC()
svc_model.fit(X_train,y_train)

#EVALUATING THE MODEL
y_predict =svc_model.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm ,annot=True)

#Model Improvisation
min_train =X_train.min()
range_train =(X_train - min_train).max()
x_train_scaled =(X_train-min_train)/range_train
sns.scatterplot(x = X_train['mean area'], y= X_train['mean smoothness'],hue =y_train)

min_test =X_test.min()
range_test =(X_test - min_test).max()
x_test_scaled =(X_test-min_test)/range_test
svc_model.fit(x_train_scaled,y_train)
y_predict =svc_model.predict(x_test_scaled)
cn = confusion_matrix(y_test,y_predict)
sns.heatmap(cn,  annot = True)

print("The accuracy of testing data: ",classification_report(y_test,y_predict))
#95% Acc