"""
Unsupervised Learning
clustering and reduction

"""

from __future__ import division #division of integers
print(__doc__)

from time import time
from scipy import stats
from scipy.spatial.distance import cdist, pdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
import operator


from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
from sklearn import mixture
from matplotlib.colors import LogNorm

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

def print_confusion_matrix(data_Y, est):
    #print(confusion_matrix(digits.target, labels))
    
    plt.imshow(confusion_matrix(data_Y, est.labels_),
               cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.grid(False)
    plt.ylabel('true')
    plt.xlabel('predicted');

def print_statistics(X, Y, name):
    samples_num, features_num = X.shape
    labels_num = len(np.unique(Y))
    print("%s:\nlabels: %d, \t samples: %d, \t features: %d\n" % (name, labels_num, samples_num, features_num))    
    
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
    
def eblow(data_X, n):
    # Determine your k range
    k_range = range(1, n+1)
    
    # Fit the kmeans model for each n_clusters = k
    k_means_var = [KMeans(n_clusters=k, n_init=20).fit(data_X) for k in k_range]
    
    # Pull out the cluster centers for each model
    centroids = [X.cluster_centers_ for X in k_means_var]
    
    # Calculate the Euclidean distance from 
    # each point to each cluster center
    k_euclid = [cdist(data_X, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke,axis=1) for ke in k_euclid]
    
    # Total within-cluster sum of squares
    wcss = [sum(d**2) for d in dist]
    
    # The total sum of squares
    tss = sum(pdist(data_X)**2)/data_X.shape[0]
    
    # The between-cluster sum of squares
    bss = tss - wcss
    
    print ("wcss:\n{0}".format(wcss))
    print ("tss:\n{0}".format(tss))
    print ("bss-tss:\n{0}".format((bss/tss)*100))    
    
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, bss/tss*100, 'b*-')
    ax.set_ylim((0,100))
    plt.grid(True)
    plt.xlabel('NUmber of clusters K')
    plt.ylabel('BSS/TSS, %')
    plt.title('Variance Explained vs. K')

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

def grid_traversal_k_attr(data_X, data_Y, attributes):
    for K in range(2, 22, 2):
        for NUM_ATTR in attributes:
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
            estimators = {"plain":[kmeans, data_X, 0],
                           "PCA":[kmeansPCA, reduced_PCA, t_PCA], 
                           "ICA":[kmeansICA, reduced_ICA, t_ICA],
                           "RP":[kmeansRP, reduced_RP, t_RP], 
                           "FA":[kmeansFA, reduced_FA, t_FA]}
            
            print(100 * '_')
            print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %   Accuracy, %     Silhouette    AUC       Time      TimeRed')
            
            out_file = "credit_kmeans_attr_K.csv"
            # Fit estimator, analize performance, and output data_X to *.csv
            fit_estimators(estimators, data_Y, out_file, NUM_ATTR, K)
            
            # Visualize the results on the reduced data
            if NUM_ATTR < 2:
                #colors = cycle('br')
                for name, est_list in estimators.items():
                    est, X, time = est_list
                    visualize(X, est, "K-means. Credit. {0}-reduced".format(name))
            
                                    
def fit_estimators(estimators, labels, filename, NUM_ATTR, K):
    datarows = []
    
    for name, est_list in estimators.items():
        estimator, data_X, time_reduction = est_list
        
        # fit the data
        t_start = time()
        estimator.fit(data_X)
        t_fit = time() - t_start
        
        silhouette = 0
        # if ground-truth labels are available
        if (((K == credit_labels_num and len(labels) == len(credit_Y)) or 
            (K == wine_labels_num and len(labels) == len(credit_Y))) and
            type(estimator) != mixture.GaussianMixture):
            #accuracy = get_accuracy(estimator, data_X, labels, name)
            accuracy = accuracy_score(credit_Y, estimator.labels_)
            auc = 0 #get_auc_score(estimator, data_X, labels)
            homogeneity = metrics.homogeneity_score(labels, estimator.labels_)
            completeness = metrics.completeness_score(labels, estimator.labels_)
            v_measure = metrics.v_measure_score(labels, estimator.labels_)
            ari = metrics.adjusted_rand_score(labels, estimator.labels_)
            ami = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
            # get clusters' sizes
            cluster0 = ((np.count_nonzero(estimator.labels_ == 0))/len(data_X))*100  
            cluster1 = ((np.count_nonzero(estimator.labels_ == 1))/len(data_X))*100
            cluster_num0 = max(cluster0, cluster1)
            cluster_num1 = min(cluster0, cluster1)
            inertia = estimator.inertia_
        else:
            accuracy, auc, homogeneity, completeness,v_measure, ari, ami = [0] *7
            cluster_num0,cluster_num1,inertia = [0] * 3
        
        bic = 0
        if type(estimator) == mixture.GaussianMixture:
            bic = estimator.bic(data_X)
        
        else:
            # silhouette is internal metrics independent of ground-truth
            silhouette = metrics.silhouette_score(data_X, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=silhouette_sample)        
        
        print('%11s      %.2f           %.2f         %.2f          %.3f     %.2f      %.2f       %.2fs        %.2fs'
          % (name, cluster_num0, cluster_num1, accuracy, silhouette, bic, auc, t_fit, time_reduction))
    
        datarows.append([name, NUM_ATTR, K, "{:.2f}".format(cluster_num0), "{:.2f}".format(cluster_num1), 
                         "{:.2f}".format(accuracy), "{:.3f}".format(silhouette), "{:.2f}".format(bic),
                         "{:.2f}".format(t_fit), "{:.2f}".format(time_reduction),
                         "{:.2E}".format(inertia), homogeneity, 
                         completeness, v_measure, ari, ami, "{:.2f}".format(auc)])
                         
    is_file = os.path.exists(filename)
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not is_file:        
            header = "Name,#Features,Clusters#,Cluster#0,Cluster#1,Accuracy,Silhouette,BIC,\
            TimeKmeans,TimeRed,inertia,homo,compl,v-means,ARI,AMI,AUC".split(",")
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

def plotGMM(est, data_X):
    # display predicted scores by the model as a contour plot
     
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    z = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    #XX = np.array([X.ravel(), Y.ravel(), z.ravel()]).T
    XX = np.linspace(np.min(data_X), np.max(data_X), 11)
    Z = -est.score_samples(XX)
    #Z = Z.reshape(X.shape)
    
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(data_X[:, 0], data_X[:, 1], .8)
    
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
        
    '''    
    delta = 0.025
    x = np.arange(-10, 10, delta)
    y = np.arange(-6, 12, delta)
    X, Y = np.meshgrid(x, y)
    #print g.means_
    plt.plot(g.means_[0][0],g.means_[0][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[1][0],g.means_[1][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[2][0],g.means_[2][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[3][0],g.means_[3][1], '+', markersize=13, mew=3)
    '''
    
def visualize(X, est, title, K):
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')#br        
        plt.figure(1)
        for k, col in zip(range(K), colors):
            my_members = est.labels_ == k
            plt.plot(X[my_members, 0], X[my_members, 1], col + 'o',  markersize=2)
            cluster_center = est.cluster_centers_[k]            
            plt.scatter(cluster_center[0], cluster_center[1], 
                        marker='^', s=169, linewidths=3,
                        color='y', zorder=10)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
        #plt.savefig("{0}_{1}_attr.jpeg".format(name, n_features))

def visualize3D(X, est, title):
    fignum = 1
    #for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1])

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Axis 1')
    ax.set_ylabel('Axis 2')
    ax.set_zlabel('Axis 3')
    plt.show()
    
def plot2D_mixture(data_X, shape, is_ax_label=False):
    
    # create fig
    plot_num = 1
    plt.figure(figsize=(16, 12))
    
    for i in range(0, shape):
        for j in range(0, shape):
            if i == j:
                ax = plt.subplot(shape, shape, plot_num)
                ax.text(0.5, 0.5,"Component " + str(i+1), 
                        {'ha': 'center', 'va': 'center'}, 
                        transform=ax.transAxes,rotation=-45)
            else:
                ax = plt.subplot(shape, shape, plot_num)
                plt.plot(data_X[:, i], data_X[:, j], 'o',  markersize=2)
                #plt.title(title)
                #plt.xlabel("Component 1")
                #plt.ylabel("Component 2")
                #plt.xticks(fontsize = 10)
                #plt.yticks(fontsize = 10)
            if not is_ax_label:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show()
            plot_num += 1
        

def get_kurtosis_skew(data_X):
    kurtosis_dict = {}
    skew_dict = {}
    for i in range(0, data_X.shape[1]):
        kurtosis = stats.kurtosis(data_X[:,i], axis=0, fisher=True, bias=True)
        skew = stats.skew(data_X[:,i], axis=0, bias=True, )
        kurtosis_dict[i] = kurtosis
        skew_dict[i] = skew
    
    return kurtosis_dict, skew_dict

def plot_distr_hist(data_X, kurtosis_list=None):
    #calc kurtosis for each feature
    if kurtosis_list is not None and len(kurtosis_list) != data_X.shape[1]:
        raise AssertionError("kurtosis_list must be equal size" )
    
    plot_num = 1
    plt.figure(figsize=(16, 12))
    
    for i in range(0, data_X.shape[1]):
        # plot distribution
        plt.subplot(6, 4, plot_num)
        kurt_data = data_X[:,i]
        plt.hist(kurt_data, 500)
        if kurtosis_list is not None:
            plt.title("Component #{0}, kurtosis={1:.2f}".format(i+1, kurtosis_list[i]), fontsize=14)
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.tight_layout()
        plt.show()
        plot_num += 1  

def find_ICA_comp(data_X, threshold):
    #t_start = time()
    reduced_ICA = FastICA(random_state=42).fit_transform(data_X)
    #time_red = time() - t_start
    
    # calculcate kurtosis and skew
    kurtosis_dict, skew_dict = get_kurtosis_skew(reduced_ICA)
   
    # sort by kurtosis in ascending order
    sorted_kurtosis = sorted(kurtosis_dict.items(), key=operator.itemgetter(1))    
    
    # create new sorted by kurtosis numpy array
    kurtosis_list = []
    reduced_ICA_sorted = np.empty(reduced_ICA.shape)
    for kurt_tuple, i in zip(sorted_kurtosis, range(reduced_ICA_sorted.shape[1])):
        j, kurtosis = kurt_tuple      
        reduced_ICA_sorted[:,i] = reduced_ICA[:,j]
        kurtosis_list.append(kurtosis)

    #print "\nKurtosis by a companent:\n{0}".format(kurtosis_list)
    #print "\nSkews by a companent:\n{0}".format(skew_list)    
    
    '''
    # select components to remove where kurtosis <= threshold
    remove_components = []
    for i in range(0, len(kurtosis_list)):
        if kurtosis_list[i] >= threshold:
            remove_components.append(i)
            
    #print remove_components
    #remove components that didn't pass threshold
    reduced_ICA_sorted = np.delete(reduced_ICA_sorted, remove_components, 1)
    '''
    
    kurtosis_list_leave = kurtosis_list[:threshold]
    reduced_ICA_leave = np.empty([reduced_ICA.shape[0], threshold])
    for i in range(threshold):
        reduced_ICA_leave[:,i] = reduced_ICA_sorted[:,i]
    
    '''
    leave = [1, 11]
    for x in range(len(reduced_ICA[0])):
        if x not in leave:
            remove_components.append(x)
    '''
    '''
    reduced_ICA_new = reduced_ICA_sorted
    kurtosis_list_new = []
    for i in range(0, len(reduced_ICA_new[0])):
        kurtosis = stats.kurtosis(reduced_ICA_new[:,i], axis=0, fisher=True, bias=True)
        kurtosis_list_new.append(kurtosis)
    
    print "\nKurtosis after removal:\n{0}".format(kurtosis_list_new) 
    '''
    return reduced_ICA_leave, kurtosis_list_leave
    
    
def find_RP_comp(data_X, data_Y, out_file):
    for K in range(2, 22, 2):
        for NUM_ATTR in attr_list_full:
            kmeansRP = KMeans(init='k-means++', n_clusters=K, n_init=50)
            
            for i in range(0, 3):
                # Generate projection
                t_start = time()
                reduced_RP = random_projection.GaussianRandomProjection(n_components=NUM_ATTR).fit_transform(data_X)
                t_RP = time() - t_start
                
                estimators = {"RP":[kmeansRP, reduced_RP, t_RP]}
                
                print(100 * '_')
                print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %   '
                    'Accuracy, %     Silhouette     BIC     AUC      Time      TimeRed')
                
                # Fit estimator, analize performance, and output data_X to *.csv
                fit_estimators(estimators, data_Y, out_file, NUM_ATTR, K)
                

def find_PCA_comp(data_X, filename):    
    pca = PCA()
    pca.fit(data_X)
    
    eigenvals = pca.explained_variance_ratio_
    #Cumulative Variance explains
    cum_varuiance = np.cumsum(np.round(eigenvals, decimals=4)*100)
    print cum_varuiance
    
    plt.plot(cum_varuiance)
    plt.ylim(20, 105)
    #plt.xlim(0, 23)
    plt.ylabel("Variance Explained %")
    plt.xlabel("Principal Comonents Number")
    plt.grid(b=True, which='major', color="#a0a0a0", linestyle='-')
    #plt.grid(b=True, which='minor', color="#a0a0a0", linestyle='-')
    plt.title("Principal Components vs. Variance")
    plt.minorticks_on()
    
    #np.savetxt(filename, [eigenvals, cum_varuiance], delimiter=",", fmt="%s")
    

def test_estimator(name, estimator, reduced_data, data_Y, time_red, NUM_ATTR, K):
    
    estimators = {name:[estimator, reduced_data, time_red]}
                
    print(100 * '_')
    print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %'  
        '  Accuracy, %     Silhouette      BIC      AUC       Time      TimeRed') 
    out_file = output_file.format(name)
    
    # Fit estimator, analize performance, and output data_X to *.csv
    fit_estimators(estimators, data_Y, out_file, NUM_ATTR, K)

def test_PCA(data_X, data_Y):
    #for K in range(2, 20, 2):
        K = 2
        for NUM_ATTR in [15]:
            # Initialize k-means objects
            kmeansPCA = KMeans(init='k-means++', n_clusters=K, n_init=50)
            
            # run Dimensionality Reduction Algorithms                        
            t_start = time()
            reduced_PCA = PCA(n_components=NUM_ATTR).fit_transform(data_X)
            t_PCA = time() - t_start
            
            #reduced_ICA = find_ICA_comp(reduced_PCA, 5)
            #plot2D_mixture(reduced_ICA, 10, False)
            # test
            test_estimator("PCA", kmeansPCA, reduced_PCA, data_Y, t_PCA, NUM_ATTR, K)
            #visualize(reduced_PCA, kmeansPCA, "K-means. PCA reduced", K)
            
            
def test_ICA(data_X, data_Y):
    for K in range(2, 16):
        #K=2
        for threshold in [100]:
            # Initialize k-means objects
            kmeans = KMeans(init='k-means++', n_clusters=K, n_init=50)
    
            # run Dimensionality Reduction Algorithms                        
            reduced_data = find_ICA_comp(data_X, threshold)
            time_red = 0.4
                  
            NUM_ATTR = len(reduced_data[0])
            # test
            test_estimator("ICA-", kmeans, reduced_data, data_Y, time_red, NUM_ATTR, K)
            
            #visualize(reduced_data, kmeans, "K-means. ICA reduced")
            #visualize3D(reduced_data, kmeans, "K-means. ICA reduced")
            #voronoi_vis(reduced_data, kmeans, 0.02, "K-means. ICA reduced")
            
def test_plain(data_X, data_Y, filename, est_name):
    NUM_ATTR = data_X.shape[0]
        
    for K in range(2, 21):
        if est_name != "gmm":
            kmeans = KMeans(init='k-means++', n_clusters=K, n_init=50)
            test_estimator(filename, kmeans, data_X, data_Y, 0, NUM_ATTR, K)
        else:
            gmm = mixture.GaussianMixture(n_components=K, covariance_type='full',
                                          max_iter=200, random_state=42, n_init=5)
            test_estimator(filename, gmm, data_X, data_Y, 0, NUM_ATTR, K)
            

def get_xy(df):
    X_raw = np.array(df.drop(df.columns[-1], 1))
    Y = np.array(df[df.columns[-1]])
    X = scale(X_raw)
    return X, Y

def combine_reduced_initial(data_reduced, data_Y, data_X=None):
    # concatenate
    if data_X is None:        
        full = np.c_[data_reduced, data_Y]
    else:
        full = np.c_[data_X, data_reduced, data_Y]
    
    # create Dataframe
    col_names = ["col" + str(x) for x in range(1, full.shape[1]+1)]
    full_df = pd.DataFrame(full, columns=col_names)                    
    return full_df

def save_to_test():
    PC_num = 13
    IC_num = 23
    
    #-----------------------------| Reduced Data [Only] |--------------------------------------#
    # ICA
    selected_ICA, kurtosis_list = find_ICA_comp(credit_X_train, IC_num)
    ICA_df = combine_reduced_initial(selected_ICA, credit_Y_train)
    ICA_df.to_csv(data_folder+'credit_ICA{0}.csv'.format(IC_num), encoding='utf-8', index=False)
    
    # PCA
    reduced_PCA = PCA(n_components=PC_num).fit_transform(credit_X_train)
    PCA_df = combine_reduced_initial(reduced_PCA, credit_Y_train)
    PCA_df.to_csv(data_folder+'credit_PCA{0}.csv'.format(PC_num), encoding='utf-8', index=False)
    
    print("\nData saved\n")
    #-----------------------------| Mixed of Raw and Reduced Data|------------------------------#
    '''
    # ICA
    selected_ICA = find_ICA_comp(credit_X_train, 2)
    ICA_df = combine_reduced_initial(credit_X_train, selected_ICA, credit_Y_train)
    ICA_df.to_csv(data_folder+'credit_ICA{0}.csv'.format(IC_num), encoding='utf-8', index=False)
    
    # PCA
    reduced_PCA = PCA(n_components=13).fit_transform(credit_X_train)
    PCA_df = combine_reduced_initial(credit_X_train, reduced_PCA, credit_Y_train)
    PCA_df.to_csv(data_folder+'credit_PCA{0}.csv'.format(PC_num), encoding='utf-8', index=False)
    '''
    
if __name__ == "__main__":
    np.random.seed(42)
    #NUM_ATTR = 2
    silhouette_sample = 15000
    attr_list_full = [2, 5, 10, 15, 20]
    output_file = "{0}.csv"
    data_folder = get_path("../../datasets/weka/")
    
    # Read in Data Sets
    credit_df = pd.read_csv(get_path("../../datasets/credit_full.csv"))
    credit_df_train = pd.read_csv(get_path("../../datasets/credit_train.csv"))
    wine_df = pd.read_csv(get_path("../../datasets/wine_full.csv"))
    wine_df_train = pd.read_csv(get_path("../../datasets/wine_train.csv"))
    
    credit_X, credit_Y = get_xy(credit_df)
    credit_X_train, credit_Y_train = get_xy(credit_df_train)
    wine_X, wine_Y = get_xy(wine_df)
    wine_X_train, wine_Y_train = get_xy(wine_df_train)
    
    # number of labels
    credit_labels_num = len(np.unique(credit_Y))
    wine_labels_num = len(np.unique(wine_Y))
    
    #Data Statistics
    print_statistics(credit_X, credit_Y, "Credit Dataset")
    print_statistics(wine_X, wine_Y, "Wine Dataset")
    
    test_plain(wine_X, wine_Y, "wine-gmm-raw", "gmm")
    #grid_traversal_k_attr(data_X)
    
    #find_PCA_comp(wine_X, "PCA-wine-var.csv")
    
    #selected_ICA, kurtosis_list = find_ICA_comp(wine_X, 11)
    #plot2D_mixture(selected_ICA, 11)
    #plot_distr_hist(selected_ICA, kurtosis_list)
    
    #reduced_RP = random_projection.GaussianRandomProjection(n_components=23).fit_transform(credit_X)
    #plot2D_mixture(reduced_RP, 10, False)
    
    #plot_distr_hist(selected_ICA)
    #plot2D_mixture(selected_ICA, 10, False)
    #eblow(credit_X, 10)
    #eblow(wine_X, 10)
    
    #find_RP_comp(data_X, "credit_kmeans_RP.csv")
    
    #test_PCA(credit_X, credit_Y)
    #test_ICA(credit_X, credit_Y)
    
    #subprocess.call(['rundll32', 'user.exe,ExitWindowsExec')
    #reduced_full_ICA = np.c_[selected_ICA, credit_Y_train]
    
    
    #---------------------------------------Test Reduced Data ----------------------------#    
    #save_to_test()
    # Conbine Reduced Data and Initial
    
    
    #reduced_full_ICA = np.c_[credit_X_train, selected_ICA, credit_Y_train]
    #np.savetxt("credit_train_ICA.csv", reduced_full_ICA, delimiter=",")
    
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
    print(pd.crosstab(kmeans.labels_, credit_Y))
    print_accuracy(kmeans, data_X, credit_Y, name='kmeans')
    print("credit_Y = 0: " + str(np.count_nonzero(credit_Y == 0)) )
    print("\n")
    
    print(pd.crosstab(kmeansPCA.labels_, credit_Y))   
    print_accuracy(kmeansPCA, reduced_PCA_unscaled, credit_Y, name='kmeansPCA')
    '''