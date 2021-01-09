# import the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("../Dataset/Breast Cancer Data.csv")
print(data.head())
print(data.shape)

print(data.columns)

# Exploratory Data Analysis
# Extracting basic information about the dataset
print(data.info())

# Determine if any null values in the data set
print(data.isna().sum())

# Heatmap indicating null values in the Dataset
plt.figure(figsize=(8,5))
plt.show(sns.heatmap(data.isnull()))

# As, we have all the 569 entries of unnamed: 32 column as NULL, We can drop that column.
data_new = data.drop(labels= 'Unnamed: 32', axis=1)
print(data_new.head())
print(data_new.info())
print(data_new.shape)

# LabelEncoding
# One column, 'diagnosis' is of type Object, which is not acceptable and thus we need to change it to int64 or float64
# First, check the no. of unique values of that column.
print(data_new['diagnosis'].unique())

sns.countplot(data_new['diagnosis'])

# We have 2 unique values 1 as M and the 2nd is B. So, we will apply the Label Encoder/One Hot Encode / get dummies function of pandas technique to convert this object datatype column to int.
le = LabelEncoder()
data_new['diagnosis'] = le.fit_transform(data_new.loc[:, 'diagnosis'])

# After encoding - 0 represents: "B" and 1 represents: "M"
print(data_new.head())
print(data_new.shape)
print(data_new.describe())

# Correlation between different features
a = data_new.corr() 
print(a)

sns.set(rc={'figure.figsize':(12,7)})
sns.heatmap(a)

# Outliers
# Now, We detect Outliers and try to remove them
z = np.abs(stats.zscore(data_new))
print(z)

threshold = 3
print(np.where(z > 3))

data_new_o = data_new[(z < 3).all(axis=1)]
print(data_new_o.shape)

# Dataset after removing outliers
print(data_new_o)

# From the above output we see that new dataframe has 487 rows which was initially 569.

# Plots
radius_info = ['radius_mean', 'radius_se', 'radius_worst']
data_new_o[radius_info].hist(layout=(2,3))

texture_info = ['texture_mean', 'texture_se', 'texture_worst'] 
data_new_o[texture_info].hist(layout=(2,3))

perimeter_info = ['perimeter_mean', 'perimeter_se', 'perimeter_worst']
data_new_o[perimeter_info].hist(layout=(2,3))

area_info = ['area_mean', 'area_se', 'area_worst']
data_new_o[area_info].hist(layout=(2,3))

smoothness_info = ['smoothness_mean', 'smoothness_se', 'smoothness_worst']
data_new_o[smoothness_info].hist(layout=(2,3))

compactness_info = ['compactness_mean', 'compactness_se', 'compactness_worst']
data_new_o[compactness_info].hist(layout=(2,3))

concavity_info = ['concavity_mean', 'concavity_se', 'concavity_worst']
data_new_o[concavity_info].hist(layout=(2,3))

concave_points_info = ['concave points_mean', 'concave points_se', 'concave points_worst']
data_new_o[concave_points_info].hist(layout=(2,3))

symmetry_info = ['symmetry_mean', 'symmetry_se', 'symmetry_worst']
data_new_o[symmetry_info].hist(layout=(2,3))

fractal_dimension_info = ['fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst']
data_new_o[fractal_dimension_info].hist(layout=(2,3))

# Finally, Storing the new data frame into a csv sheet
data_new_o.to_csv('Cleaned_data_Breast_Cancer.csv', index=False)
