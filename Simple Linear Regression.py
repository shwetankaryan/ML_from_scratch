import pandas as pd
import numpy as np

#dummy data with one independent variable.
x = np.random.randn(100)
#Dependent variable y is linearly dependent on x
y = 6.01 *x + 12.91

# #Prediction
# def pred(x, w, b):
#     return x*w + b

# #Loss function
# def loss(x,w,b, y):
#     return np.sum(1/2 * (pred(x, w, b) - y)**2)

# #Gradient 
# def gradients(x,w,b, y):
#     dw = np.sum((pred(x,w,b) - y) * x)
#     db = np.sum(pred(x, w, b) - y)

#     return dw, db

# def fit (x, y, w, b, lr,  iterations):
#     for i in range(iterations):
#         l = np.round(loss(x, w,b, y), 2)
#         print (i, w, b, l)
#         dw  = gradients(x, w, b, y) [0]
#         db = gradients(x, w, b, y) [1]

#         w -= dw*lr
#         b -= db*lr

# fit(x = x, y = y, w = 0.1, b = 0.2, lr = 0.01 ,  iterations = 50)

class LinearRegression:

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def pred(self, x):
        return x * self.w + self.b

    def loss(self, x, y):
        return 1/2 * np.mean((self.pred(x) - y)**2)
    
    def gradients(self, x,y):
        dw = np.mean((self.pred(x) - y) * x)
        db = np.mean(self.pred(x) - y)

        return dw, db
    
    def fit (self, x, y, lr, iterations ):
        for i in range(iterations):
            dw, db = self.gradients(x, y)

            print (np.round(self.w, 3), np.round(self.b, 3), np.round (self.loss(x, y)))

            self.w -= dw * lr
            self.b -= db * lr

initial_w = 0.1
initial_b = -10.9
model = LinearRegression(initial_w, initial_b)
model.fit(x, y, 0.1, 100)
