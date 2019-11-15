import pandas as pd
import numpy as np
import time
import sys
 
columns = ['latitude', 'longitude', 'reviewCount', 'checkins']
 
class Cluster():
    def __init__(self, neighbors, values):
        self.neighbors = neighbors
        self.values = values
        
#def compare(clusters_i, clusters_j):
#        for index, row in clusters_i.iterrows():
            
##    print(clusters_i)
##    for i in range(len(clusters_i)):
#        for column in columns:
#            print()
#            if abs(np.subtract(clusters_i.values[column] - clusters_j.values[column])) > 0.00001:
#                return False
#            return True
 
def within_cluster_score(clusters, dist_type='euchlidean'):
    square_err = 0
    for cluster in clusters:
        for row, index in cluster.neighbors.iterrows():
            if dist_type.strip().lower() == 'manhattan':
                square_err += np.square(np.sum(abs(np.subtract(row, cluster.values))))
            else:
                square_err += np.square(np.linalg.norm(row - cluster.values))
    return square_err

#minimize the within cluster sum of squares
#maximize the distance between centroids
def improved_scoring(clusters):
    dist = 0
    for cluster_i in clusters:
        for cluster_j in clusters:
            dist += np.square(np.linalg.norm(cluster_i.values - cluster_j.values))
    return within_cluster_score(clusters) - dist
 
def euchlidean(data_row, cluster_row):
    return np.linalg.norm(data_row - cluster_row)

def manhattan(cluster_row, data_row):
    return np.sum(abs(np.subtract(data_row, cluster_row)))
    
def KNN(k_val, data, iterations, dist_type='euchlidean'):
    #chooses k clusters from the data (randomly)
    k_clusters = data.sample(n = k_val)
    compare_clusters = pd.DataFrame(columns=columns)
    clusters = []
    
    start = time.time()
    while not compare_clusters.equals(k_clusters):#compare(compare_clusters, k_clusters):
#        print('iteration')
        #create a Cluster object for each random cluster
        clusters = []
#        print(k_clusters)
        for index, row in k_clusters.iterrows():
            clusters.append(Cluster(pd.DataFrame(columns=columns), row))
            
        for cluster in clusters:
            cluster.neighbors = cluster.neighbors.append(cluster.values)
#            print(cluster.neighbors)
#            print(cluster.values)
        
#        #for every index (except clusters), find it's nearest neighbor
#        for row in pd.concat([data, k_clusters, k_clusters]).drop_duplicates(keep=False).itertuples(index=False):
#            
#            tup = np.array((row.latitude, row.longitude, row.reviewCount, row.checkins))
#            
#            #adds a column which computes the of the test point to evey cluster
#            if dist_type.strip().lower() == 'manhattan':
#                k_clusters['distance'] = k_clusters.apply(lambda x: manhattan(x, tup), axis=1)
#            else:
##                if iterations > 100:
##                    print(k_clusters.apply(lambda x: euchlidean(x, tup), axis=1))
#                k_clusters['distance'] = k_clusters.apply(lambda x: euchlidean(x, tup), axis=1)
#            min_index = k_clusters['distance'].idxmin()
#            k_clusters.drop(['distance'], axis=1, inplace=True)
##            print(min_index)
#            min_row = k_clusters.loc[min_index]
#            mins = int(np.where(k_clusters.index==min_index)[0])
#            
##            print(clusters[mins].neighbors)
##            print('min_row: ', min_row.drop(labels=['distance']))
##            print(clusters[mins].neighbors)
#            clusters[mins].neighbors = clusters[mins].neighbors.append(min_row, ignore_index=True)

    
        #for every index (except clusters), find it's nearest neighbor
        for index, row in pd.concat([data, k_clusters, k_clusters]).drop_duplicates(keep=False).iterrows():
            closest_cluster = None
            shortest_dist = np.inf
#            tup = np.array((row.latitude, row.longitude, row.reviewCount, row.checkins))
            #test the difference to each cluster and select the closest
            for cluster in clusters:
            
                if dist_type.strip().lower() == 'manhattan':
                    dist = np.sum(abs(np.subtract(row, cluster.values)))
                else:
                   #calculate L2 (euchlidean) distance as default
                    dist = np.linalg.norm(row - cluster.values)
#                    print(dist)
            
                if dist <= shortest_dist:
                    shortest_dist = dist
                    closest_cluster = cluster
 
#            print('before: ', closest_cluster.neighbors)
#            index = list(columns)
            closest_cluster.neighbors = closest_cluster.neighbors.append(row)
#            print('after: ', closest_cluster.neighbors)
            
        #clears the DataFrame, since these will not be the clusters any more
#        k_clusters.round(4)
        compare_clusters = k_clusters.copy()
        k_clusters = k_clusters.iloc[0:0]
#        print('before: ', k_clusters)
        #calculate the mean for each column
        for cluster in clusters:
            for column in columns:
                #calculate new centroid coordinates for each of the four columns
                #include current centroid values into calculation as well
                if len(cluster.neighbors) > 0:
#                    print(column)
#                    print('col: ', cluster.values[column])
#                    print('neighbors: ', cluster.neighbors)
#                    print('neighbors[column]: ', cluster.neighbors[column])
#                    print('sum: ', np.sum(cluster.neighbors[column]))
#                    print('length: ', len(cluster.neighbors))
                    cluster.values[column] = np.mean(cluster.neighbors[column])#float(np.sum(cluster.neighbors[column])) / float(len(cluster.neighbors))
                else:
#                    print(cluster.neighbors)
#                    print(cluster.values)
                    pass
            k_clusters = k_clusters.append(cluster.values)
#            k_clusters.round(4)
#        print('after: ', k_clusters)
#        if not iterations % 10:
#            print(k_clusters)
        iterations += 1
        #just in case
        if start - time.time() > 590:
            break
            
    print('iterations: ', iterations)
    end = time.time()
    print(end - start)
    
#    for cluster in clusters:
#        print(cluster.neighbors)
#        print(cluster.values)
#        print()
        
#        print(euchlidean(cluster.neighbors, cluster.values))
    
    return clusters
 
 
if __name__ == '__main__':

    
    train_file = '..\data\given\\test.csv'#sys.argv[1]
    k_val = 4#int(sys.argv[2])
    cluster = 1#int(sys.argv[3])
    iterations = 1
    file = open("data.txt", "a")
    data = pd.read_csv(train_file, delimiter=',', dtype=float, usecols=columns, index_col=None, engine='python')
    sampled_data = data.sample(n = int(np.floor((6 * data.shape[0]) / 100)))
    for cluster in [1, 2, 3, 4, 5, 6]:
        print('clustering option: -----', cluster)
        file.write(str('clustering option: -----' +  str(cluster)) + '\n')
        for k_val in [3, 6, 9, 12, 24]:
            print('k_value: =====', k_val)
            file.write('k_value: ====='+ str(k_val) + '\n')

            centroids = None
            if cluster is 1: #regular euchlidean distance
                centroids = KNN(k_val, data, iterations)
#                print('WC-SSE=', within_cluster_score(centroids))
                file.write('WC-SSE='+ str(within_cluster_score(centroids)) + '\n')
            elif cluster is 2: #log transform
                log_data = data.copy()
                log_data['reviewCount'] = np.log2(data['reviewCount'])
                log_data['checkins'] = np.log2(data['checkins'])
                centroids = KNN(k_val, log_data, iterations)
#                print('WC-SSE=', within_cluster_score(centroids))
                file.write('WC-SSE='+ str(within_cluster_score(centroids)) + '\n')
            elif cluster is 3: #standardize
                std_data = data.copy()
                for column in columns:
                    std_data[column] = (data[column] - data[column].mean()) / data[column].std()
                centroids = KNN(k_val, std_data, iterations)
#                print('WC-SSE=', within_cluster_score(centroids))
                file.write('WC-SSE='+ str(within_cluster_score(centroids)) + '\n')
            elif cluster is 4: #manhattan distance
                centroids = KNN(k_val, data, iterations, 'manhattan')
#                print('WC-SSE=', within_cluster_score(centroids, 'manhattan'))
                file.write('WC-SSE='+ str(within_cluster_score(centroids, 'manhattan')) + '\n')
            elif cluster is 5:  #random sample of data
                centroids = KNN(k_val, sampled_data, iterations)
#                print('WC-SSE=', within_cluster_score(centroids))
                file.write('WC-SSE='+ str(within_cluster_score(centroids)) + '\n')
            elif cluster is 6: #improved score function
                centroids = KNN(k_val, data, iterations)
#                print('WC-SSE=', improved_scoring(centroids))
                file.write('WC-SSE='+ str(improved_scoring(centroids)) + '\n')
         
            k = 1
            for centroid in centroids:
                print('Centroid', k, '=[',
                centroid.values['latitude'], centroid.values['longitude'],
                centroid.values['reviewCount'], centroid.values['checkins'], ']')
                k += 1
                
             
                file.write('Centroid' + str(k) + '=[' +
                    str(centroid.values['latitude'])     + str(centroid.values['longitude']) + 
                    str(centroid.values['reviewCount'])  + str(centroid.values['checkins']) +  ']' + '\n') 
    file.close() 
