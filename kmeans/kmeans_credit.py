"""

http://fromdatawithlove.thegovans.us/2013/05/clustering-using-scikit-learn.html
"""

from __future__ import division
print(__doc__)

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import random_projection



def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    return abs_file_path

# http://www.caner.io/purity-in-python.html
def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]
    
def get_accuracy(clf, X, Y, name='clf'):
    correct = 0    
    pred_array = clf.predict(X)
    for i in range(len(X)):
        if pred_array[i] == Y[i]:
            correct += 1
    
    # since algorithm labels clusters  randomly, choose the largest cluster
    accuracy1 = correct/len(X)*100
    accuracy2 = 100 - accuracy1
    #accuracy = "Accuracy_{0}: {1:.2f}%".format(name, max(accuracy1, accuracy2))
    return max(accuracy1, accuracy2)
    

def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('% 11s   %.2fs    %.2E   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_)))
             #metrics.silhouette_score(data, estimator.labels_,
             #                         metric='euclidean',
             #                         sample_size=sample_size)))
                                      
def print_clusters(classifiers):
    for name, clf_list in classifiers.items():
        clf, data_used = clf_list
        #print("\n" + name + "\n" + str(40 * '_') + "\n")
        cluster0 = ((np.count_nonzero(clf.labels_ == 0))/len(data_used))*100  
        cluster1 = ((np.count_nonzero(clf.labels_ == 1))/len(data_used))*100
        #print("cluster_kmean #0: {0:.2f}%\ncluster_kmean #1: {1:.2f}%".format(cluster0, cluster1))
        accuracy = get_accuracy(clf, data_used, data_Y, name)
    
        print('%11s      %.2f           %.2f         %.2f         %.3f'
          % (name, cluster0, cluster1, accuracy, 
             metrics.silhouette_score(data_used, clf.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
                                      

def visualize(reduced_data, clf, h, title):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    #h = 0.02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = clf.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    train_df = pd.read_csv(get_path("../../datasets/credit_full.csv"))
    
    # convert the "quality" label column to numpy arrays
    data_X = np.array(train_df.drop(['LIMIT_BAL','DEFAULT', 'SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], 1))
    #data_X = np.array(train_df.drop(['DEFAULT'], 1))
    data_Y = np.array(train_df['DEFAULT'])
    
    # Scaling
    scaler = StandardScaler()
    #data_X_scaled = scaler.fit_transform(data_X)    
    data_X_scaled = scale(data_X)
    
    #credit
    n_samples, n_features = data_X.shape
    n_creditY = len(np.unique(data_Y))
    #n_creditY = 5
    sample_size = 10000
    
    print("n_creditY: %d, \t n_samples: %d, \t n_features: %d"
      % (n_creditY, n_samples, n_features))
    
    kmeans = KMeans(init='k-means++', n_clusters=n_creditY, n_init=50)
    kmeans_scaled = KMeans(init='k-means++', n_clusters=n_creditY, n_init=50)
    kmeansPCA = KMeans(init='k-means++', n_clusters=n_creditY, n_init=50)
    kmeansPCA_scaled = KMeans(init='k-means++', n_clusters=n_creditY, n_init=50)
    
    # run PCA
    reduced_data = PCA(n_components=5).fit_transform(data_X)
    reduced_scaled = PCA(n_components=5).fit_transform(data_X_scaled)
    
    # Randomized Projections 
    X = np.random.rand(100, 10000)
    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)
    
    classifiers = {"default":[kmeans, data_X],"scaled":[kmeans_scaled, data_X_scaled], 
              "PCA-default":[kmeansPCA, reduced_data], "PCA-scaled":[kmeansPCA_scaled, reduced_scaled]}
    
    # Generate clusters
    print(79 * '_')
    print('% 10s' % 'init      '
          '    time      inertia   homo    compl  v-meas     ARI     AMI  silhouette')
    for name, clf_list in classifiers.items():     
        clf, data_used = clf_list
        bench_k_means(clf, name, data_used, data_Y)
    print(79 * '_')
    
    # Analize clusters size
    cluster0_true = ((np.count_nonzero(data_Y == 0))/len(data_Y))*100 
    cluster1_true = ((np.count_nonzero(data_Y == 1))/len(data_Y))*100 
    print("cluster_true #0: {0:.2f}%\ncluster_true #1: {1:.2f}%\n".format(cluster0_true, cluster1_true))
    print(79 * '_')
    print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %   Accuracy, %     silhouette')
    
    # print statistics
    print_clusters(classifiers)
    
    # Visualize the results on PCA-reduced data
    #visualize(reduced_scaled, kmeansPCA_scaled, 0.02, "K-means on the Credit PCA-reduced data (scaled)")
    #visualize(reduced_data, kmeansPCA, 1000, "K-means on the Credit PCA-reduced data (unscaled)")
    
    
    #http://brandonrose.org/clustering
    #http://stackoverflow.com/questions/32930647
    #similarities = metrics.euclidean_distances(reduced_scaled)
    #reduced_mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1).fit(similarities).embedding_#.fit_transform(similarities)
    #pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    
    reduced_mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1).fit(similarities).embedding_#.fit_transform(similarities)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    
    '''
    # Confusion Matrix
    print(pd.crosstab(kmeans.labels_, data_Y))
    print_accuracy(kmeans, data_X, data_Y, name='kmeans')
    print("data_Y = 0: " + str(np.count_nonzero(data_Y == 0)) )
    print("\n")
    
    print(pd.crosstab(kmeansPCA.labels_, data_Y))   
    print_accuracy(kmeansPCA, reduced_data, data_Y, name='kmeansPCA')
    '''
    ###############################################################################
    
    '''
    for i in set(kmeans.labels_):
        index = kmeans.labels_ == i
        plt.plot(data_X[index,0], data_X[index,1], 'o')
    plt.show()
    '''