#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling1D,BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import layers,models

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing the dataset
dataset=pd.read_csv('../Dataset/Cleaned_data_Breast_Cancer.csv')

print(dataset.shape)

x1 = dataset.drop(['id','diagnosis'],axis =1)
X = pd.DataFrame(data = x1, columns=x1.columns)
y = np.array(dataset.diagnosis)
target_names = np.array(['malignant','benign'])

#Splitting into Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#reshaping the X dataset from (r,c) to (r,c,1)
X_train = X_train.reshape(389,30,1)
X_test = X_test.reshape(98, 30, 1)

X_test.shape

epochs = 45

#Creating the model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape = (30,1)))
model.add(MaxPooling1D(pool_size=4))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))
#model summary
model.summary()

model.compile(optimizer=Adam(learning_rate=0.00005), loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1,)
#model.save("../cnn_model.h5")
#testing over training dataset to create a confusuion matrix
prediction = model.predict(X_train)
#using 0.5 as a threshold value
prediction_array = (prediction>0.5)

results = model.evaluate(X_test, y_test)
# accuracy: 0.969
print(model.predict(X_test) > 0.5).astype("int32")