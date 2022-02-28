
import pandas as pd
import numpy as np

data  = pd.read_csv("train.csv")
#print(data.head(5))

m, n = data.shape
#print(m,n)

data = np.array(data)
#print(data)

#Random shuffle
np.random.shuffle(data)

#Division de Dev y Train
data_dev = data[0:1000].T
X_dev = data_dev[1:n]
Y_dev = data_dev[0]

data_train = data[1000:m].T
X_train = data_train[1:n]
Y_train = data_train[0]

#Iniciar parametros
def iniciar_parametros():
    w1 = np.random.rand(10,784)
    b1 = np.random.rand(10,1)
    
    w2 = np.random.rand(10,10)
    b2 = np.random.rand(10,1)
    
    print(w1.shape)
    print(b1)
    return w1, b1, w2, b2
w1, b1, w2, b2 = iniciar_parametros()


def ReLU(x):
    return np.maximum(0,x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def forward_prop(w1,w2,b1,b2,X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

z1, a1, z2, a2 = forward_prop(w1,w2,b1,b2,X_dev)

print(a2.shape)