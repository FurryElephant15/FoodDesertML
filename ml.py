import numpy as np
from tensorflow import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense        
from imblearn.over_sampling import RandomOverSampler                       
from sklearn.model_selection import train_test_split  # imports improts imports!!!
with open("FoodAccessResearchAtlasData2019CSVREV1.csv", "r", encoding="utf-8-sig") as f:
    data = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)

x=data[:, 0:15]
y=data[:, 15] #defining data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #manually splitting the data takes too long, lets let scikit do it for me (this better work i spent the better part of a day trying to install scikit learn. Moral: FUCK conda)
ros = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
y_train_resampled = keras.utils.to_categorical(y_train_resampled, num_classes=2)

model = Sequential() #define model
model.add(Dense(15,input_dim=15, activation='relu' )) #defo subject to change soon
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #compile dat
model.fit(x_train_resampled,y_train_resampled, epochs = 50, batch_size = 10)  #train dat
_, accuracy = model.evaluate(x_test, keras.utils.to_categorical(y_test, num_classes=2))
print('Accuracy: %.2f' % (accuracy*100)) #please 70% first try????? a man can dream ahahahaha
