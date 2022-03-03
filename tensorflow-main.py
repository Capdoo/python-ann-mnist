import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline

data_train = pd.read_csv("train.csv")

#data = np.toarray(data_train)

X = data_train.iloc[:,1:785].values
y = data_train.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

## Paises
columnTransformer = ColumnTransformer([("0", OneHotEncoder(), [0])], remainder='passthrough')
y = columnTransformer.fit_transform(y)
## Eliminando columna de más
#y = y[:, 1:]



# 3. Division dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





import keras
from keras.models import Sequential
from keras.layers import Dense

##Secuential para iniciar la red neuronal
classifier = Sequential()

##Añadir la capa de entrada y oculta
classifier.add(Dense(units = 16, kernel_initializer = "uniform", 
                     activation = "relu", input_dim = 784))

## Añadir la segunda capa oculta
classifier.add(Dense(units = 16, kernel_initializer = "uniform", 
                     activation = "relu"))

##Añadir capa de salida
classifier.add(Dense(units = 10, kernel_initializer = "uniform", 
                     activation = "sigmoid"))


## Compilar la red neuronal

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


### Ajustar la ANN con el entrenamiento
#batch_size = 10
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)


# 5. Evaluación de la ANN
#y_pred = classifier.predict(X_test)

## Usando un umbral
#y_pred = (y_pred > 0.5)


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

print((1548+134)/2000)





