import pandas as pd
import numpy as np
import sys

 
columns = ['latitude', 'longitude', 'reviewCount', 'checkins']
 
class Cluster():
    def __init__(self, neighbors, values):
        self.neighbors = neighbors
        self.values = values
 
def within_cluster_score(clusters, dist_type='euchlidean'):
    square_err = 0
    for cluster in clusters:
        for row, index in cluster.neighbors.iterrows():
            if dist_type.strip().lower() == 'manhattan':
                square_err += np.square(np.sum(abs(np.subtract(row, cluster.values))))
            else:
                square_err += np.square(np.sum(np.linalg.norm(row - cluster.values)))
    return square_err

def improved_scoring(clusters):
    pass
 
def KNN(k_val, data, iterations, dist_type='euchlidean'):
    #chooses k clusters from the data (randomly)
    k_clusters = data.sample(n = k_val)
    clusters = []
    terminate = False
    
    print('k_clusters: ', k_clusters)
 
    while not terminate:
 
        #create a Cluster object for each random cluster
        #O(k)
#        print('k_clusters: ', k_clusters)
        clusters = []
        for index, row in k_clusters.iterrows():
            clusters.append(Cluster(pd.DataFrame(columns=columns), row))
            
            
 
#        print('k_clusters: ', k_clusters)
#        print('difference: ', pd.concat([data, k_clusters, k_clusters]).drop_duplicates(keep=False))
    
        #for every index (except clusters), find it's nearest neighbor
        for index, row in pd.concat([data, k_clusters, k_clusters]).drop_duplicates(keep=False).iterrows():
#            print('here')
            closest_cluster = None
            shortest_dist = np.inf
 
            #test the difference to each cluster and select the closest
            for cluster in clusters:
 
                if dist_type.strip().lower() == 'manhattan':
                    dist = np.sum(abs(np.subtract(row, cluster.values)))
                else:
                   #calculate L2 (euchlidean) distance as default
                    dist = np.linalg.norm(row - cluster.values)
 
                if dist <= shortest_dist:
                    shortest_dist = dist
                    closest_cluster = cluster
 
            closest_cluster.neighbors = closest_cluster.neighbors.append(row)
 
        
    
        #clears the DataFrame, since these will not be the clusters any more
        compare_clusters = k_clusters.copy()
        k_clusters = k_clusters.iloc[0:0]
#        print('k_clusters: ', k_clusters)
#        print(clusters)
        #calculate the mean for each column
        for cluster in clusters:
#            print('here')
#            print(cluster.neighbors)
            for column in columns:
                #calculate new centroid coordinates for each of the four columns
                #include current centroid values into calculation as well
                
                if len(cluster.neighbors) > 0:
#                    print('before: ', cluster.values[column])
                    cluster.values[column] = (np.sum(cluster.neighbors[column]) + cluster.values[column]) / len(cluster.neighbors)
#                    print('after: ', cluster.values[column])
            k_clusters = k_clusters.append(cluster.values)
#            print('k_clusters: ', k_clusters)
#            
#        print('k_clusters: ', k_clusters)
#        print('clusters: ', clusters)
        
#        print('compare_clusters: ', compare_clusters)
#        print('k_clusters: ', k_clusters)
#        print()
        
        #either nothing has changed or the iterations have run out
        if compare_clusters.equals(k_clusters) or iterations is 0:
            terminate = True
#            print(iterations)
        iterations += 1
    print(iterations)
    return clusters
 
 
if __name__ == '__main__':
 
    train_file = '..\data\given\\test.csv'#sys.argv[1]
    k_val = 3#sys.argv[2]
    cluster = 1#sys.argv[3]
    iterations = 1
 
    data = pd.read_csv(train_file, delimiter=',', dtype=float, usecols=columns, index_col=None, engine='python')
#    print(data)
#    for k_val in [3, 6, 9, 12, 24]:
#        pass
 
    centroids = None
 
    if cluster is 1:
        #regular euchlidean distance
        centroids = KNN(k_val, data, iterations)
        print('WC-SSE=', within_cluster_score(centroids))
    elif cluster is 2:
        #log transform
        data['reviewCount'] = np.log2(data['reviewCount'])
        data['checkins'] = np.log2(data['checkins'])
        centroids = KNN(k_val, data, iterations)
        print('WC-SSE=', within_cluster_score(centroids))
    elif cluster is 3:
        #standardize
        #x_std = (x - mean) / standard_deviation
        for column in columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        centroids = KNN(k_val, data, iterations)
        print('WC-SSE=', within_cluster_score(centroids))
    elif cluster is 4:
        #manhattan distance
        centroids = KNN(k_val, data, iterations, 'manhattan')
        print('WC-SSE=', within_cluster_score(centroids, 'manhattan'))
    elif cluster is 5:
        #random sample of data
        sampled_data = data.sample(n = int(np.floor((6 * data.shape[0]) / 100)))
        centroids = KNN(k_val, data, iterations)
        print('WC-SSE=', within_cluster_score(centroids))
    elif cluster is 6:
        #improved score function
        centroids = KNN(k_val, data, iterations)
        print('WC-SSE=', improved_scoring(centroids))
 
    k = 0
    for centroid in centroids:
        print('Centroid', k, '=[',
        centroid.values['latitude'], centroid.values['longitude'],
        centroid.values['reviewCount'], centroid.values['checkins'], ']')
        k += 1
