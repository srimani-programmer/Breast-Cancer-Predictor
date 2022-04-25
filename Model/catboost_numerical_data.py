# import subprocess
# import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'catboost'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier


dataset = pd.read_csv('Breast Cancer Data.csv')

#print(dataset.info) --> checking nan values
# col 33 nan values so we'll ignore it.
x = dataset.iloc[:, 2:32].values
# Mapping 2 categories to values
y = dataset["diagnosis"].replace({"M": 0, "B":1})

# Normalizing the data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)


print(x.shape)

# Check nan if there's missing values.
# print(np.isnan(x))

# 30% Test
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,shuffle=True, random_state=0)

# Classifier Model_1
# l2_leaf_reg: Coefficient at the L2 regularization term of the cost function
# rsm: Random subspace method. The percentage of features to use at each split selection.
clf = CatBoostClassifier(iterations=80,learning_rate = 0.03, eval_metric="Accuracy", devices='0:1', max_depth=8 ,l2_leaf_reg=0.8,rsm=0.4)
clf.fit(xtrain,ytrain,eval_set = (xtest, ytest),verbose = True)
# bestTest = 0.9824561404
# bestIteration = 39

plot_confusion_matrix(clf, xtest, ytest) 

plt.show()

#Model_2
#Increasing training dataset samples --> 20% Test
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,shuffle=True, random_state=0)

clf = CatBoostClassifier(iterations=110,learning_rate = 0.07, eval_metric="Accuracy", devices='0:1', max_depth=5 ,l2_leaf_reg=0.8,rsm=0.4)
clf.fit(xtrain,ytrain,eval_set = (xtest, ytest),verbose = True)
# bestTest = 0.9912280702
# bestIteration = 96

plot_confusion_matrix(clf, xtest, ytest) 

plt.show()