import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def prediction(self, X):
        return np.dot(X, self.w) + self.b
        
    def loss(self, X, y):
        return np.sum(0.5 * (y - (np.dot(X, self.w) + self.b))**2 )/len(y)
        
    def dldw(self, X, y):
        return - np.dot((y - (np.dot(X, self.w) + self.b)) ,  X)/len(y)
        
    def dldb(self, X, y):
        return - np.sum(y - (np.dot(X, self.w) + self.b))/len(y)
        
    def fit(self,X , y, epochs = 5000, batch_size = 50, lr = 0.001):
        for i in range(epochs):
            
            indices = np.random.choice(len(X), batch_size, replace=False)
            
            X_ = X[indices]
            y_ = y[indices]
            
            dldw_ = self.dldw(X_, y_)
            dldb_ = self.dldb(X_, y_)
            
            loss = np.round(self.loss(X_, y_), 2)
            if i % 500 == 0:
                print (f'Epochs {i}/{epochs},  loss {loss}')
#             print (np.round(loss, 1))
            
            self.w = self.w - dldw_ * lr
            self.b = self.b - dldb_ * lr

#Checking with 1 independent variable 
X = np.random.randn(1000)
y = 5.1 * X + 3

X = X.reshape(-1, 1)

# Initialize model parameters
initial_w = np.random.randn(1)
initial_b = np.random.randn(1)
initial_w, initial_b

model = LinearRegression(initial_w, initial_b)

model.fit(X, y)

print(model.w, model.b)