import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import plotly.express as px
import time

# Set the random seed for reproducibility
np.random.seed(42)

# Number of points per cluster
n_points = 10000

# Generate random points for the first cluster
cluster1 = np.random.randn(n_points//2, 2) + np.array([5, 10])

# Generate random points for the second cluster
cluster2 = np.random.randn(n_points//2, 2) + np.array([1, 5])

# Generate random points for the second cluster
cluster3 = np.random.randn(n_points//2, 2) + np.array([10, 15])

# Combine the clusters to form the dataset
data = np.vstack((cluster1, cluster2, cluster3))

# # Plot the data
plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', edgecolor='k')
plt.title('2D Data with Three Clusters')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

class KMeans:
    
    def __init__(self, c, n_centroid):
        self.n_centroid = n_centroid
        self.centroids = c
        
    def distance(self, p):
        return [np.sqrt(np.mean((x-y)**2)) for x in p for y in self.centroids]
    
    def fit(self, X, iteration = 100):
        
        df = pd.DataFrame(X)
        
        x_columns = ['x' + str(i) for i in range(X.shape[1])]
        centroid_cols = ['centroid_' + str(i) for i in range(self.n_centroid)]
        
        df.columns = x_columns
        
        
        for i in range(iteration):
            distances = []
            for p in range(len(df)):
                
                point = df.iloc[p, 0:X.shape[-1]].values
                point = point.reshape(1, point.shape[0])
                distances.append(self.distance(point))

            distances = np.array(distances)
            for c_col in range(len(centroid_cols)) :
                df[centroid_cols[c_col]] = np.array(distances)[:, c_col]
            
            df['new centroid'] = df[centroid_cols].idxmin(axis = 1)
            
            fig = px.scatter(df, x='x0', y='x1', color='new centroid')
            fig.show()
            clear_output(wait=True)
            time.sleep(3)
            
            new_centroid = []
            for c in df['new centroid'].unique():
                new_centroid.append(np.mean(df[df['new centroid'] == c][x_columns], axis = 0).tolist())

            new_centroids = np.array(new_centroid)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return df
    
b = 10
a = 1
initial_centroid = np.random.rand(3,2) * (b - a) + a

model = KMeans(initial_centroid, 3)

result_df = model.fit(data, iteration = 5)