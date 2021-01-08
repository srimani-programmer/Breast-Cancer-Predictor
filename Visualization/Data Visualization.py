import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
cancer = pd.read_csv('Breast Cancer Data.csv')
X = cancer.iloc[:, 2:32].values
y = cancer.iloc[:, 1].values
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))

#VISUALISING THE DATA
sns.pairplot(df_cancer , vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension'])

sns.pairplot(df_cancer ,hue ='target', vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity'])


sns.countplot(df_cancer['target'])
sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data =df_cancer)
plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)