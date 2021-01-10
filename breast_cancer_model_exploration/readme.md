### VARIOUS CLASSIFICATION MODEL COMPARISION AND CHOOSING THE BEST ONE.
I have compared 5 classification models using grid search and displayed the results in a python notebook with visualistaion. After some parameter and kernel tuning support vector classifier turned out to be the best model for predicting with about 97.62 percent accuracy on test set. So SVC can be used to predict. I used rbf kernel in SVC. reference file: gridsearch.py

Along with that i used Artificial neural network to train the model and it gave me about 99 percent accuracy on test set. The model was saved in .h5 format to be used for backend implementation. reference files: breast_cancer_nn.py

### VISUALIZATION FOR GRIDSEARCH
![alt text](https://github.com/spursbyte/Breast-Cancer-Predictor/blob/New_Pipeline/breast_cancer_model_exploration/box_plot.png)
