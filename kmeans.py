import pandas as pd
import numpy as np
import sys
import time
import math
from scipy.spatial.distance import pdist


class Cluster():
    def __init__(self, neighbors, values):
        self.neighbors = neighbors
        self.values = values
        
def calc_centroid():
    pass
        
if __name__ == '__main__':
    
    train_file = '..\data\given\\test.csv'#sys.argv[1]
    k_val = 4#sys.argv[2]
    cluster = 1#sys.argv[3]
    
    columns = ['latitude', 'longitude', 'reviewCount', 'checkins']
    
    data = pd.read_csv(train_file, delimiter=',', dtype=float, usecols=columns, index_col=None, engine='python')
    
    #chooses k clusters from the data (randomly)
    k_clusters = data.sample(n = k_val)
    clusters = []
    
    #create a Cluster object for each random cluster
    for index, row in k_clusters.iterrows():
        clusters.append(Cluster(pd.DataFrame(columns=columns), row))
        
        
    print(len(data))
    #for every index, find it's nearest neighbor
    for index, row, in data.iterrows():
        closest_cluster = None
        shortest_dist = test = math.inf
        
        #test the difference to each cluster and select the closest
        for cluster in clusters:
            
            #skip if it's the same datapoint
            if row.equals(cluster.values):
                continue
            
            dist = np.linalg.norm(row - cluster.values)
            
            if dist <= shortest_dist:
                shortest_dist = dist
                closest_cluster = cluster
                
        closest_cluster.neighbors = closest_cluster.neighbors.append(row)
        
        
#    print(clusters[0].neighbors.head())
#    
#    print(len(clusters[0].neighbors))
#    print(len(clusters[1].neighbors))
#    print(len(clusters[2].neighbors))
#    print(len(clusters[3].neighbors))
    
    
    for cluster in clusters:
        print('old: ', cluster.values)
        for column in columns:
            #calculate new centroid coordinates for each of the four columns
            #include current centroid values into calculation as well
            cluster.values[column] = (np.sum(cluster.neighbors[column]) + cluster.values[column]) / len(cluster.neighbors)
        print('new: ', cluster.values)
        print()
    
    
    #merged should always be an empty dataframe, so that every data point is assigned
    #to a unique cluster point
#    merged = pd.merge(clusters[2].neighbors, clusters[3].neighbors, on=columns, how='inner')
#    print('merged: ', merged)
    
    