"""
Unsupervised Learning
clustering and reduction

"""

from __future__ import division #division of integers
print(__doc__)

from time import time
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import random_projection
from itertools import cycle
from sklearn.metrics import roc_auc_score

def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    return abs_file_path
    
def get_accuracy(clf, X, Y, name='clf'):
    correct = 0    
    pred_array = clf.predict(X)
    for i in range(len(X)):
        if pred_array[i] == Y[i]:
            correct += 1
    
    # since algorithm labels clusters  randomly, choose the largest cluster
    accuracy1 = correct/len(X)*100
    accuracy2 = 100 - accuracy1
    return max(accuracy1, accuracy2)

def get_auc_score(clf, X, Y):   
    pred_Y = clf.predict(X)
    auc = roc_auc_score(Y, pred_Y)
    # since algorithm labels clusters  randomly, choose the largest cluster
    return auc

def plot_variance_retained(data_X):
    pca = PCA()
    pca.fit(data_X)
    
    #evector = pca.components_
    evals = pca.explained_variance_
    evals_ratio = pca.explained_variance_ratio_
    
    #Cumulative Variance explains
    cum_variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print "PCA. Number of componenets selection"
    print(80 * '_')
    print "Variance explained:\n{0}\nEigenvalues:\n{1}\nEigenvalues_ratio:\n{2}".format(cum_variance, evals, evals_ratio)
    
    plt.plot(cum_variance)
    plt.ylim(20, 105)
    plt.xlim(0, 23)
    plt.ylabel("Variance Explained %")
    plt.xlabel("Principal Comonents Number")
    plt.grid(b=True, which='major', color="#a0a0a0", linestyle='-')
    #plt.grid(b=True, which='minor', color="#a0a0a0", linestyle='-')
    plt.title("Principal Components vs. Variance")
    plt.minorticks_on()
      
    
def plot_K_vs_Silhouette(data_X, k_list):
    # set Number of Cluster
    s = []
    for n_clusters in k_list:
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=50)
        kmeans.fit(data_X)
        
        #append silhouette_score
        s.append(metrics.silhouette_score(data_X, kmeans.labels_, metric='euclidean', sample_size=silhouette_sample))
    
    # plot the result
    plt.plot(k_list, s)
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Clusters K")
    plt.title("Silhouette vs. K")
    plt.show()

def bench_k_means(estimator, name, data_X, labels):
    t_start = time()
    estimator.fit(data_X)
    t_spent = time() - t_start
    
    homogeneity = metrics.homogeneity_score(labels, estimator.labels_)
    completeness = metrics.completeness_score(labels, estimator.labels_)
    v_measure = metrics.v_measure_score(labels, estimator.labels_)
    ari = metrics.adjusted_rand_score(labels, estimator.labels_)
    ami = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)    
    
    print('% 11s   %.2fs    %.2E   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name, t_spent, estimator.inertia_, homogeneity,
             completeness, v_measure, ari, ami))
             #metrics.silhouette_score(data_X, estimator.labels_,
             #                         metric='euclidean',
             #                         sample_size=sample_size)))

def grid_traversal_k_attr():
    for K in range(2, 22, 2):
        for NUM_ATTR in attr_list_full:
            # Initialize k-means objects
            kmeans = KMeans(init='k-means++', n_clusters=K, n_init=50)
            kmeansPCA = KMeans(init='k-means++', n_clusters=K, n_init=50)
            kmeansICA = KMeans(init='k-means++', n_clusters=K, n_init=50)
            kmeansRP = KMeans(init='k-means++', n_clusters=K, n_init=50)
            kmeansFA = KMeans(init='k-means++', n_clusters=K, n_init=50)
            
            # run Dimensionality Reduction Algorithms                        
            t_start = time()
            reduced_PCA = PCA(n_components=NUM_ATTR).fit_transform(data_X)
            t_PCA = time() - t_start
            
            t_start = time()
            reduced_ICA = FastICA(n_components=NUM_ATTR, random_state=42).fit_transform(data_X)
            t_ICA = time() - t_start
            
            t_start = time()
            reduced_RP = random_projection.GaussianRandomProjection(n_components=NUM_ATTR).fit_transform(data_X)
            t_RP = time() - t_start
            
            t_start = time() #http://scikit-learn.org/stable/modules/decomposition.html#factor-analysis
            reduced_FA = FactorAnalysis(n_components=NUM_ATTR, random_state=37, max_iter=3000).fit_transform(data_X)
            t_FA = time() - t_start
            
            # Dict of list containing: kmeans object, reduced data_X, time for reduction 
            classifiers = {"plain":[kmeans, data_X, 0],
                           "PCA":[kmeansPCA, reduced_PCA, t_PCA], 
                           "ICA":[kmeansICA, reduced_ICA, t_ICA],
                           "RP":[kmeansRP, reduced_RP, t_RP], 
                           "FA":[kmeansFA, reduced_FA, t_FA]}
            
            print(100 * '_')
            print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %   Accuracy, %     Silhouette    AUC       Time      TimeRed')
            
            out_file = "credit_kmeans_attr_K.csv"
            # Fit estimator, analize performance, and output data_X to *.csv
            fit_estimators(classifiers, data_Y, out_file, NUM_ATTR, K)
            
            # Visualize the results on the reduced data
            if NUM_ATTR < 2:
                #colors = cycle('br')
                for name, est_list in classifiers.items():
                    est, X, time = est_list
                    visualize(X, est, "K-means. Credit. {0}-reduced".format(name))
                                      
def fit_estimators(estimators, labels, filename, NUM_ATTR, K):
    datarows = []
    for name, est_list in estimators.items():
        estimator, data_used, time_reduction = est_list
        
        # fit the data
        t_start = time()
        estimator.fit(data_used)
        t_fit = time() - t_start
        
        # get clusters' sizes
        cluster0 = ((np.count_nonzero(estimator.labels_ == 0))/len(data_used))*100  
        cluster1 = ((np.count_nonzero(estimator.labels_ == 1))/len(data_used))*100
        cluster_num0 = max(cluster0, cluster1)
        cluster_num1 = min(cluster0, cluster1)
        
        accuracy = get_accuracy(estimator, data_used, labels, name)
        auc = get_auc_score(estimator, data_used, labels)
        
        homogeneity = metrics.homogeneity_score(labels, estimator.labels_)
        completeness = metrics.completeness_score(labels, estimator.labels_)
        v_measure = metrics.v_measure_score(labels, estimator.labels_)
        ari = metrics.adjusted_rand_score(labels, estimator.labels_)
        ami = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)  
        silhouette = metrics.silhouette_score(data_used, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=silhouette_sample)        
        
        print('%11s      %.2f           %.2f         %.2f          %.3f        %.2f       %.2fs        %.2fs'
          % (name, cluster_num0, cluster_num1, accuracy, silhouette, auc, t_fit, time_reduction))
    
        datarows.append([name, NUM_ATTR, K, "{:.2f}".format(cluster_num0), "{:.2f}".format(cluster_num1), 
                         "{:.2f}".format(accuracy), "{:.3f}".format(silhouette), "{:.2f}".format(t_fit), 
                         "{:.2f}".format(time_reduction),"{:.2E}".format(estimator.inertia_), homogeneity, 
                         completeness, v_measure, ari, ami, "{:.2f}".format(auc)])
                         
    is_file = os.path.exists(filename)
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not is_file:        
            header = "Name,#Features,Clusters#,Cluster#0,Cluster#1,Accuracy,Silhouette,\
            TimeKmeans,TimeRed,inertia,homo,compl,v-meas,ARI,AMI,AUC".split(",")
            writer.writerow(header)    
        writer.writerows(datarows)
                                      

def voronoi_vis(X, est, h, title):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    #h = 0.02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = est.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = est.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
def visualize(X, est, title): 
        colors = cycle('br')        
        plt.figure()
        for k, col in zip(range(n_creditY), colors):
            my_members = est.labels_ == k
            plt.plot(X[my_members, 0], X[my_members, 1], col + 'o',  markersize=2)
                #cluster_center = est.cluster_centers_[k]            
            #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #         markeredgecolor='k', markersize=14)
        plt.title(title)
        plt.show()
        #plt.savefig("{0}_{1}_attr.jpeg".format(name, n_features))

def find_ICA_comp():
    reduced_ICA = FastICA(random_state=42).fit_transform(data_X)
    
    #calc kurtosis for each feature
    kurtosis_list = []
    for i in range(0, len(reduced_ICA[0])):
        kurtosis = stats.kurtosis(reduced_ICA[:,i], axis=0, fisher=True, bias=True)
        kurtosis_list.append(kurtosis)
    
    print "\nKurtosis by a companent:\n{0}".format(kurtosis_list)
    
    # select components to remove where kurtosis < 100   
    remove_components = []
    for i in range(0, len(kurtosis_list)):
        if abs(kurtosis_list[i]) <= 100:
            #selected_ICA.append(reduced_ICA[:,0])
            remove_components.append(i)
    
    # remove components that didn't pass threshold
    reduced_ICA_new = np.delete(reduced_ICA, remove_components, 1)
        
    kurtosis_list_new = []
    for i in range(0, len(reduced_ICA_new[0])):
        kurtosis = stats.kurtosis(reduced_ICA_new[:,i], axis=0, fisher=True, bias=True)
        kurtosis_list_new.append(kurtosis)
    
    print "\nKurtosis after removal:\n{0}".format(kurtosis_list_new) 
    return reduced_ICA_new
    
def find_RP_comp():
    pass

def find_PCA_comp():    
    pca = PCA()
    pca.fit(data_X)
    
    #Cumulative Variance explains
    cum_varuiance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print cum_varuiance
    
    plt.plot(cum_varuiance)
    plt.ylim(20, 105)
    plt.xlim(0, 23)
    plt.ylabel("Variance Explained %")
    plt.xlabel("Principal Comonents Number")
    plt.grid(b=True, which='major', color="#a0a0a0", linestyle='-')
    #plt.grid(b=True, which='minor', color="#a0a0a0", linestyle='-')
    plt.title("Principal Components vs. Variance")
    plt.minorticks_on()          

if __name__ == "__main__":
    np.random.seed(42)
    NUM_ATTR = 2
    silhouette_sample = 15000
    attr_list_full = [2, 5, 10, 15, 20]
    #attr_list_reduced = [2, 6, 10]
    #output_file = "credit_kmeans_23attr.csv"
    
    full_df = pd.read_csv(get_path("../../datasets/credit_full.csv"))
    small_df = pd.read_csv(get_path("../../datasets/credit_test_num.csv"))
    
    # 30% of dataset
    #data_X = np.array(small_df.drop(['LIMIT_BAL'], 1))
    #data_Y = np.array(small_df['DEFAULT'])
    
    # convert the "quality" label column to numpy arrays
    #data_X = np.array(full_df.drop(['LIMIT_BAL','DEFAULT', 'SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], 1))
    data_X_raw = np.array(full_df.drop(['DEFAULT'], 1))
    data_Y = np.array(full_df['DEFAULT'])
    
    # Scaling
    #data_X_scaled = StandardScaler().fit_transform(data_X)   
    data_X_scaled = scale(data_X_raw)
    #data_X_scaled = data_X 
    data_X = data_X_scaled
    
    #Data Statistics
    n_samples, n_features = data_X_raw.shape
    n_creditY = len(np.unique(data_Y))
    #n_creditY =
    cluster0_true = ((np.count_nonzero(data_Y == 0))/len(data_Y))*100 
    cluster1_true = ((np.count_nonzero(data_Y == 1))/len(data_Y))*100 
    print("cluster_true #0: {0:.2f}%\ncluster_true #1: {1:.2f}%\n".format(cluster0_true, cluster1_true))
    print("n_creditY: %d, \t n_samples: %d, \t n_features: %d" % (n_creditY, n_samples, n_features))
    
    
    #grid_traversal_k_attr()
    
    #find_PCA_comp()
    selected_ICA = find_ICA_comp()
    
    

    
    #######################################################################################################
    '''    
    if NUM_ATTR < 3:
        voronoi_vis(reduced_PCA_scaled, kmeansPCA_scaled, 0.02, "K-means on the Credit PCA-reduced data (scaled)")
        voronoi_vis(reduced_PCA_unscaled, kmeansPCA, 1000, "K-means on the Credit PCA-reduced data (unscaled)")
        voronoi_vis(reduced_ICA, kmeansICA, 0.005, "K-means on the Credit ICA-reduced data")
        voronoi_vis(reduced_PCA_scaled, kmeansPCA_scaled, 0.02, "K-means on the Credit FA-reduced data (scaled)")
        voronoi_vis(reduced_RP, kmeansRP, 0.02, "K-means on the Credit RP-reduced data")
        voronoi_vis(reduced_FA, kmeansFA, 0.02, "K-means on the Credit FA-reduced data")
    '''
    
    '''
    # Subplots
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    plot_num = 1
    plt.figure(figsize=(16, 12))    
    
    for name, clf_list in classifiers.items():
        clf, X, time = clf_list
        #if hasattr(clf, 'labels_'):
        y_pred = clf.labels_.astype(np.int)
        #else:
        #    y_pred = clf.predict(X)
        
        # plot
        plt.subplot(3, 3, plot_num)
        #if i_dataset == 0:
        plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=2)
        
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        #plt.xlim(x_min, x_max)
        #plt.ylim(y_min, y_max)
        #plt.xlim(-2, 2)
        #plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        #plt.text(.99, .01, ('%.2fs' % (0.01)).lstrip('0'),
        #         transform=plt.gca().transAxes, size=15,
        #         horizontalalignment='right')
        plot_num += 1

    plt.show()
    '''    
    
    '''
    for i in set(kmeans.labels_):
        index = kmeans.labels_ == i
        plt.plot(data_X[index,0], data_X[index,1], 'o')
    plt.show()
    '''
    
    '''
    #a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
    #np.savetxt("foo.csv", a, delimiter=",")
    '''
    
    '''
    # Confusion Matrix
    print(pd.crosstab(kmeans.labels_, data_Y))
    print_accuracy(kmeans, data_X, data_Y, name='kmeans')
    print("data_Y = 0: " + str(np.count_nonzero(data_Y == 0)) )
    print("\n")
    
    print(pd.crosstab(kmeansPCA.labels_, data_Y))   
    print_accuracy(kmeansPCA, reduced_PCA_unscaled, data_Y, name='kmeansPCA')
    '''