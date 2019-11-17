import numpy as np
import pandas as pd
import sys

def within_cluster_score(clusters, k_clusters, dist_type='euclidean'):
    square_err = 0
    for index, neighbor in k_clusters.items():
        for item in neighbor:
            if dist_type.strip().lower() == 'manhattan':
                square_err += np.square(np.sum(abs(np.subtract(item, clusters[index]))))
            else:
                square_err += np.square(np.linalg.norm(np.subtract(item, clusters[index])))
    return square_err

#minimize the within cluster sum of squares & maximize the distance between centroids
def improved_scoring(clusters, k_clusters, dist_type='euclidean'):
    dist = 0
    for index_i, cluster_i in clusters.items():
        
        for index_j, cluster_j in clusters.items():
            if dist_type.strip().lower() == 'manhattan':
                dist += np.square(np.sum(abs(np.subtract(cluster_i, cluster_j))))
            else:
                dist += np.square(np.linalg.norm(np.subtract(cluster_i, cluster_j)))
    return within_cluster_score(clusters, k_clusters) - dist

def KNN(k_val, data, converge_val = 0.00000001, dist_type='euclidean'):
    
    #select random indexes from the data
    rands = np.random.choice(len(data), k_val, replace=False)
    clusters = {}
    k = 0
    for rand in rands:      #fill in the clusters with initial data points
        clusters[k] = data[rand]
        k += 1
        
    ret_clusters = {}
    optimal = False
    #while the values are not close enough
    while not optimal:
        k_clusters = {}
        
        for i in range(k_val):
            k_clusters[i] = []
            
        for row in data:
            dist = []
            for cluster in clusters:
                if dist_type.strip().lower() == 'manhattan':
                    dist.append(np.sum(abs(np.subtract(row, clusters[cluster]))))
                else:
                    dist.append(np.linalg.norm(np.subtract(row, clusters[cluster])))
            
            shortest = dist.index(min(dist))
            k_clusters[shortest].append(row)
            
        compare_cluster = dict(clusters)

        #compute new clusters
        for neighbor in k_clusters:
            clusters[neighbor] = np.average(k_clusters[neighbor], axis = 0)
            
        for cluster in clusters:
            old = compare_cluster[cluster]
            new = clusters[cluster]
            
            if np.sum((new - old)/old * 100.0) > converge_val:
                optimal = True
        ret_clusters = k_clusters
        
    return clusters, ret_clusters
            
if __name__ == "__main__":
    
    train_file = sys.argv[1]
    k_val = int(sys.argv[2])
    cluster_type = int(sys.argv[3])

    #treat it as a numpy array, NOT pandas
    data = pd.read_csv(train_file, delimiter=',', 
                       dtype=float, 
                       usecols=['latitude', 'longitude', 'reviewCount', 'checkins'], 
                       index_col=None, 
                       engine='python')
    log_data = data.copy()
    std_data = data.copy()
    sampled_data = data.sample(n = int(np.floor((6 * data.shape[0]) / 100))).values
    data = data.values
    
    if cluster_type is 1: #regular euclidean distance
        clusters, k_clusters = KNN(k_val, data)
        print('WC-SSE=', within_cluster_score(clusters, k_clusters))
    elif cluster_type is 2: #log transform
        log_data['reviewCount'] = np.where(log_data['reviewCount'] != 0, np.log2(log_data['reviewCount']), 0)
        log_data['checkins'] = np.log2(log_data['checkins'])
        clusters, k_clusters = KNN(k_val, log_data.values)
        print('WC-SSE=', within_cluster_score(clusters, k_clusters))
    elif cluster_type is 3: #standardize
        for column in std_data.columns:
            std_data[column] = (std_data[column] - std_data[column].mean()) / std_data[column].std()
        clusters, k_clusters = KNN(k_val, std_data.values)
        print('WC-SSE=', within_cluster_score(clusters, k_clusters))
    elif cluster_type is 4: #manhattan distance
        clusters, k_clusters = KNN(k_val, data)
        print('WC-SSE=', within_cluster_score(clusters, k_clusters, dist_type='manhattan'))
    elif cluster_type is 5:  #random sample of data
        clusters, k_clusters = KNN(k_val, sampled_data)
        print('WC-SSE=', within_cluster_score(clusters, k_clusters))
    elif cluster_type is 6: #improved score function
        clusters, k_clusters = KNN(k_val, data)
        print('WC-SSE=', improved_scoring(clusters, k_clusters))
    
    k = 1
    for index, cluster in clusters.items():
        print('Centroid' + str(k) + '=' + str(list(cluster)))
        k += 1
