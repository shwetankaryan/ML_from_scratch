import pandas as pd
import numpy as np

#Dummy data with one independent variable.
x = np.random.randn(100)
#Dependent variable y is linearly dependent on x
y = 6.01 *x + 12.91 + np.random.randn(100)/1000

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
