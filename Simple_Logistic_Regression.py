import pandas as pd
import numpy as np

a,b = 1,5
x1 = np.random.rand(1000) * (b - a) + a
y1 = 0

c,d = 10, 15
x2 = np.random.rand(1000) * (d - c) + c
y2 = 1

df1 = pd.DataFrame({'x' : x1, 'y' : y1})
df2 = pd.DataFrame({'x' : x2, 'y' : y2})
df = pd.concat([df1, df2])

df = df.sample(frac=1).reset_index(drop=True)
df.head()

#Logistic Regression

# y -> sigmoid(wx + b)
x = df['x'].values
y = df['y'].values

class LogisticRegression:
    
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
    def prediction(self, x):
        regress = self.w*x + self.b
        pred = self.sigmoid(regress)
        return pred
    
    def loss(self, x, y):
        pred = self.prediction(x)
        loss_ = - np.mean(y* np.log(pred) + (1-y)*np.log(1 - pred))
        return loss_
    
    def dldw(self, x, y):
        pred = self.prediction(x)

        dloss_dpred = pred - y
        dpred_dregress = pred * (1 - pred)
        dregress_dw = x

        dldw = np.mean(dloss_dpred * dpred_dregress * dregress_dw)
        return dldw

    def dldb(self, x, y):
        pred = self.prediction(x)

        dloss_dpred = pred - y
        dpred_dregress = pred * (1 - pred)
        dregress_db = 1

        dldb = np.mean(dloss_dpred * dpred_dregress * dregress_db)
        return dldb
    
    def fit(self, x, y, iteration = 10000, batch_size = 100):
        losses = []
        for i in range(iteration):
            indices = np.random.choice(len(x), batch_size, replace=False)
            x_ = x[indices]
            y_ = y[indices]
            
            dldw_ = self.dldw(x_, y_)
            dldb_ = self.dldb(x_,y_)
            loss = np.round(self.loss(x_, y_), 4) 
            losses.append(loss)
            if i % 500 == 0:
                print (f'Epochs {i}/{iteration}, loss {loss}')

            self.w -= dldw_ * 0.01
            self.b -= dldb_ * 0.01

initial_w = np.random.randn(1)
initial_b = np.random.randn(1)

model = LogisticRegression(w = initial_w, b= initial_b)
model.fit(x = x, y = y)
model.w, model.b