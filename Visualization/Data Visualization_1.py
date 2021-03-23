#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
cancer = pd.read_csv('../Dataset/Breast Cancer Data.csv')
data = pd.DataFrame(data = cancer, columns=cancer.columns)

#dropping the last unnamed 32 column
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
#dataset information
data.info()

diagnosis_all = list(data.shape)[0]
#counting the total records of benign and malignant
diagnosis_categories = list(data['diagnosis'].value_counts())
print("The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all, 
                                                                      diagnosis_categories[0], 
                                                                      diagnosis_categories[1]))
features_mean= list(data.columns[1:11])


menu = '''
-------------------------------------------
Enter your choice for data visualization
-------------------------------------------
1. Heat Map
2. Scatter Matrix
3. Distribution Plot of each type of diagnosis for each of the mean features.
4. Box Plot
5. Exit
--------------------------------------------
'''

while True:
    print(menu)
    n = int(input("Enter Your Choice : "))
    if n ==1:
        #heatmap
        plt.figure(figsize=(10,10))
        sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
        plt.show()
    elif n == 2:
        #scatter matrix
        color_dic = {'M':'red', 'B':'blue'}
        colors = data['diagnosis'].map(lambda x: color_dic.get(x))

        sm = pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));

        plt.show()
    elif n == 3:
        #distplot
        bins = 12
        plt.figure(figsize=(15,15))
        for i, feature in enumerate(features_mean):
            rows = int(len(features_mean)/2)
    
            plt.subplot(rows, 2, i+1)
    
            sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');
            sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');
    
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
    
    elif n == 4:
        #boxplot
        plt.figure(figsize=(15,15))
        for i, feature in enumerate(features_mean):
            rows = int(len(features_mean)/2)
            
            plt.subplot(rows, 2, i+1)
    
            sns.boxplot(x='diagnosis', y=feature, data=data, palette="Set1")

        plt.tight_layout()
        plt.show()
    else:
        break