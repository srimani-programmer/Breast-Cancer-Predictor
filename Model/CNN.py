#Importing Libraries

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def plot_learningCurve(history, epoch):
  '''
    A function to plot the accuracy and loss curve of a model over epochs

    
    '''
  plt.subplot(1, 2, 1)
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  

  # Plot training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()


#importing the csv file
cancer = pd.read_csv('../Dataset/Cleaned_data_Breast_Cancer.csv')
x1 = cancer.drop(['id','diagnosis'],axis =1)
X = pd.DataFrame(data = x1, columns=x1.columns)
y = np.array(cancer.diagnosis)
target_names = np.array(['malignant','benign'])

#spliting the dataset into train and test dataset with 0.8 to 0.2 ratio respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#reshaping the X dataset from (r,c) to (r,c,1)
X_train = X_train.reshape(389,30,1)
X_test = X_test.reshape(98, 30, 1)

#total epochs
epochs = 50

#Creating the model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape = (30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
#model summary
model.summary()

#using adam optimizer with learning rate of 0.00005 and loss function as binary crossentropy
model.compile(optimizer=Adam(lr=0.00005), loss = 'binary_crossentropy', metrics=['accuracy'])

#adding a checkpointer to save the model weights if the accuracy improves over validation dataset
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

#training the model
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1, callbacks=[checkpointer])

#loading model with best weights over validation dataset
model.load_weights('model.weights.best.hdf5')

#plotting the curve for accuracy and loss function
plot_learningCurve(history, epochs)

#testing over training dataset to create a confusuion matrix
prediction = model.predict(X_train)
#using 0.5 as a threshold value
prediction_array = (prediction>0.5)
confusion_matrix(y_train, prediction_array)
'''
The confusion matrix is :
    
array([[259,   2],
       [  3, 125]], dtype=int64)
'''
results = model.evaluate(X_test, y_test)
# accuracy: 0.9796