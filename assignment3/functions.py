"""
Goncharenko Nikolay 
CS 7641: Machine Learning

Assignment #3
Unsupervised Learning and 
Dimensionality Reduction
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.learning_curve import learning_curve

def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    return abs_file_path

def create_folder(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    
def get_accuracy(clf, X, Y, name='clf'):
    correct = 0    
    pred_array = clf.predict(X)
    for i in range(len(X)):
        if pred_array[i] == Y[i]:
            correct += 1
    
    # since algorithm assigns clusters id randomly,  
    # choose the largest cluster
    accuracy1 = correct/len(X)*100
    accuracy2 = 100 - accuracy1
    return max(accuracy1, accuracy2)

def get_auc_score(clf, X, Y):   
    pred_Y = clf.predict(X)
    auc = roc_auc_score(Y, pred_Y)
    return auc

def plot_df_matrix(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    pd.scatter_matrix(df_norm, alpha = 0.3, figsize = (30,30), diagonal = 'kde');

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
    k_means_var = [KMeans(n_clusters=k, n_init=50).fit(data_X) for k in k_range]
    
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
    plt.xlabel('Number of clusters K')
    plt.ylabel('BSS/TSS, %')
    plt.title('Variance Explained vs. K')

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt, train_scores_mean, test_scores_mean

def fit_svm(data_X, data_Y):
    data_Y = data_Y.astype(str)    
    cv = cross_validation.StratifiedShuffleSplit(data_Y, n_iter=10,test_size=0.2, random_state=42)
    gamma, C = 0.06, 2
    clf = SVC(gamma=gamma, C=C).fit(data_X, data_Y)
    cv_score = training_score(clf, data_X, data_Y, cv=cv)
    cv_mean = np.array(cv_score).mean()
    print "cv_score: {0:2f}".format(cv_mean)
    #heldout_score = clf.score(test_X, test_Y)
    return cv_mean
    
                    
def print_cluster_sizes(estimator, data_X, data_Y, K):
    # get labels
    if type(estimator) == mixture.GaussianMixture:
        est_labels = estimator.predict(data_X)
    else:
        est_labels = estimator.labels_            
    
    cluster_distr = []
    for i in range(K):
        cluster_distr.append(((np.count_nonzero(est_labels == i))/len(data_X))*100)
    print cluster_distr
    
    print(pd.crosstab(est_labels, data_Y))
            
def get_estimator(est_name, K):
    if "gmm" not in est_name:
        est = KMeans(init='k-means++', n_clusters=K, n_init=50)
    else:
        est = mixture.GaussianMixture(n_components=K, covariance_type='full',
                                  max_iter=200, random_state=42, n_init=5)
    return est
                                    
def fit_estimators(estimators, data_Y, K):
    datarows = []
    for name, est_list in estimators.items():
        estimator, data_X, time_reduction = est_list
        
        # fit the data
        t_start = time()
        estimator.fit(data_X)
        t_fit = time() - t_start

        silhouette, bic, inertia, auc = [0]*4
        NUM_ATTR = data_X.shape[1]
        header = "Name,#Features,Clusters#,Cluster#0,Cluster#1,Accuracy,Silhouette,BIC,\
        TimeKmeans,TimeRed,inertia,homo,compl,v-means,ARI,AMI,AUC".split(",")
        
        #cv = cross_validation.StratifiedShuffleSplit(credit_Y, n_iter=5,test_size=0.2, random_state=42)
        # if ground-truth data_Y are available
        if ((K == credit_labels_num and len(data_Y) == len(credit_Y)) or 
            (K == wine_labels_num and len(data_Y) == len(wine_Y))):
            if type(estimator) == mixture.GaussianMixture:
                est_labels = estimator.predict(data_X)
            else:
                est_labels = estimator.labels_
            accuracy = get_accuracy(estimator, data_X, data_Y, name)
            accuracy = accuracy_score(data_X, est_labels)
            #accuracy = cross_val_score(estimator, data_X, credit_Y, cv=cv, scoring='accuracy')
            #auc = cross_val_score(estimator, data_X, credit_Y, cv=cv, scoring='roc_auc')
            #auc = get_auc_score(estimator, data_X, data_Y)
            homogeneity = metrics.homogeneity_score(data_Y, est_labels)
            completeness = metrics.completeness_score(data_Y, est_labels)
            v_measure = metrics.v_measure_score(data_Y, est_labels)
            ari = metrics.adjusted_rand_score(data_Y, est_labels)
            ami = metrics.adjusted_mutual_info_score(data_Y,  est_labels)
            # get clusters' sizes
            cluster0 = ((np.count_nonzero(est_labels == 0))/len(data_X))*100  
            cluster1 = ((np.count_nonzero(est_labels == 1))/len(data_X))*100
            cluster_num0 = max(cluster0, cluster1)
            cluster_num1 = min(cluster0, cluster1)
            if type(estimator) != mixture.GaussianMixture:
                inertia = estimator.inertia_
        else:
            accuracy, auc, homogeneity, completeness,v_measure, ari, ami = [0]*7
            cluster_num0,cluster_num1,inertia = [0]*3
        
        filename = ""
        if type(estimator) == mixture.GaussianMixture:
            filename = "{0}.csv".format(gmm_folder + name)            
            bic = estimator.bic(data_X)
            # get silhouette score
            em_labels = estimator.predict(data_X)
            silhouette = metrics.silhouette_score(data_X, em_labels,
                                          metric='euclidean',
                                          sample_size=silhouette_sample)
                                         
            datarows.append([name, NUM_ATTR, K, "{:.2f}".format(bic),
                             "{:.3f}".format(silhouette), "{:.2f}".format(t_fit), 
                             "{:.2f}".format(time_reduction),
                             "{:.2f}".format(accuracy),"{:.2f}".format(auc),
                             "{:.2f}".format(cluster_num0)])
            header = "Name,#Features,Clusters#,BIC,Silhouette,TimeKmeans,TimeRed,Accuracy,AUC,Cluster1".split(",")
            #print_cluster_sizes(estimator, data_X, data_Y, K)            
            #print(pd.crosstab(est_labels, credit_Y))

        else:
            # silhouette is internal metrics independent of ground-truth
            filename = "{0}.csv".format(km_folder + name)
            silhouette = metrics.silhouette_score(data_X, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=silhouette_sample)
                                          
            datarows.append([name, NUM_ATTR, K, "{:.2f}".format(cluster_num0), "{:.2f}".format(cluster_num1), 
                         "{:.2f}".format(accuracy), "{:.3f}".format(silhouette), "{:.2f}".format(bic),
                         "{:.2f}".format(t_fit), "{:.2f}".format(time_reduction),
                         "{:.2E}".format(inertia), homogeneity, 
                         completeness, v_measure, ari, ami, "{:.2f}".format(auc)])
                
        print('%11s      %.2f           %.2f         %.2f          %.3f     %.2f      %.2f       %.2fs        %.2fs'
          % (name, cluster_num0, cluster_num1, accuracy, silhouette, bic, auc, t_fit, time_reduction))
    
    
    # save the result                     
    is_file = os.path.exists(filename)
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not is_file:      
            writer.writerow(header)    
        writer.writerows(datarows)

    
def visualize(X, est, title, K):        
        
        if type(est) == mixture.GaussianMixture:
           labels = est.predict(X)
           cluster_centers = est.means_
        else:   
            labels = est.labels_        
            cluster_centers = est.cluster_centers_
        
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')#br    
        plt.figure(1)
        for k, col in zip(range(K), colors):
            my_members = labels == k
            plt.plot(X[my_members, 0], X[my_members, 1], col + 'o',  markersize=2)
            cluster_center = cluster_centers[k]            
            plt.scatter(cluster_center[0], cluster_center[1], 
                        marker='^', s=169, linewidths=3,
                        color='y', zorder=10)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
        #plt.savefig("{0}_{1}_attr.jpeg".format(name, n_features))

def visualize3D(X, est, title, K=None):
    if type(est) == mixture.GaussianMixture:
           labels = est.predict(X)
           
    else:   
        labels = est.labels_ 
    
    fignum = 1
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1])
    plt.cla()
       
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


def slice_ICA_data(data_X, kurtosis_list, num_comp_keep):
    # num_comp_keep - a number of sorted components:
    # leave all negative + add 'num_comp_keep - negative comp' 
    #num of largest in descending order
    if num_comp_keep < 2:
        negative_kurt_num = 1
    else:
        negative_kurt_num = len([x for x in kurtosis_list if x < 0])
    # accounts for first negative_kurt_num elements    
    highest_num = num_comp_keep - negative_kurt_num 
    if highest_num <= 0:
        sliced_kurtosis_list = kurtosis_list[:negative_kurt_num]
        sliced_data = data_X[:,:negative_kurt_num]
    else:
        sliced_kurtosis_list = kurtosis_list[:negative_kurt_num] + kurtosis_list[-highest_num:]
        sliced_data = np.c_[data_X[:,:negative_kurt_num],data_X[:,-highest_num:]]  
    return sliced_data, sliced_kurtosis_list


def slice_ICA_highest(data_X, kurtosis_list, num_comp_keep):
    # accounts for first negative_kurt_num elements     
    sliced_kurtosis_list = kurtosis_list[-num_comp_keep:]
    sliced_data = data_X[:,-num_comp_keep:]
    return sliced_data, sliced_kurtosis_list

def slice_ICA_list(data_X, kurtosis_list, list_elem):
    # accounts for first negative_kurt_num elements     
    sliced_kurtosis_list = [kurtosis_list[i] for i in list_elem]
    sliced_data = data_X[:,list_elem]
    return sliced_data, sliced_kurtosis_list    

def get_reduced_ICA(data_X, num_comp_keep):
    selected_ICA, kurtosis_list, est = find_ICA_comp(data_X)
    sliced_ICA_data, sliced_kurt = slice_ICA_data(data_X, kurtosis_list, num_comp_keep)
    return sliced_ICA_data, sliced_kurt


def find_ICA_comp(data_X):
      
    ica = FastICA(random_state=42)
    #t_start = time()    
    reduced_ICA = ica.fit_transform(data_X)
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
    return reduced_ICA_sorted, kurtosis_list, ica
  

def find_PCA_comp(data_X, filename):    
    pca = PCA()
    pca.fit(data_X)
    
    eigenvals = pca.explained_variance_ratio_
    #Cumulative Variance Explained
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
    

def test_estimator(name, estimator, reduced_data, data_Y, time_red, K):
    
    estimators = {name:[estimator, reduced_data, time_red]}
                
    print(100 * '_')
    print('% 10s' % 'Features' '    Cluster #0, %     Cluster #1, %'  
        '  Accuracy, %     Silhouette      BIC      AUC       Time      TimeRed') 
    
    # Fit estimator, analize performance, and output data_X to *.csv
    fit_estimators(estimators, data_Y, K)


# Factor Analysis
#http://scikit-learn.org/stable/modules/decomposition.html#factor-analysis          
def find_FA(data_X, data_Y, filename, est_name):
    for NUM_ATTR in range(1, data_X.shape[1]+1):
        fa = FactorAnalysis(n_components=NUM_ATTR, random_state=37, max_iter=3000)
        reduced_FA = fa.fit_transform(data_X)            
        select_comp_supervised(reduced_FA, data_Y, filename, NUM_ATTR, est_name)
                

# Randomized projection          
def find_RP(data_X, data_Y, filename, est_name):
    for NUM_ATTR in range(1, data_X.shape[1]+1):
        for i in range(5):
            rp = random_projection.GaussianRandomProjection(n_components=NUM_ATTR)
            reduced_RP = rp.fit_transform(data_X)
            select_comp_supervised(reduced_RP, data_Y, filename, NUM_ATTR, est_name)
            #select_comp_supervised(data_X, data_Y, filename, 11, est_name)


def find_PCA(data_X, data_Y, filename, est_name):
    for NUM_ATTR in range(11, data_X.shape[1]+1):
            reduced_PCA = PCA(n_components=NUM_ATTR).fit_transform(data_X)
            select_comp_supervised(reduced_PCA, data_Y, filename, NUM_ATTR, est_name)

def find_ICA(data_X, data_Y, filename, est_name):
    reduced_data, kurt_list, ica = find_ICA_comp(data_X)    
    for NUM_ATTR in range(1, data_X.shape[1]+1):
            sliced_ICA_data, sliced_kurt = slice_ICA_data(reduced_data, kurt_list, NUM_ATTR) 
            #sliced_ICA_data, sliced_kurt = slice_ICA_highest(reduced_data, kurt_list, NUM_ATTR)            
            select_comp_supervised(sliced_ICA_data, data_Y, filename, NUM_ATTR, est_name)


def test_FA(data_X, data_Y, filename, est_name, NUM_ATTR):
    for K in range(10, 11):
            # Initialize objects
            est = get_estimator(est_name, K)
            
            # run Dimensionality Reduction Algorithms                        
            t_start = time()
            fa = FactorAnalysis(n_components=NUM_ATTR, random_state=42, max_iter=3000)
            reduced_FA = fa.fit_transform(data_X) 
            t_FA = time() - t_start
            # test
            #eblow(reduced_PCA, 20)
            test_estimator(filename, est, reduced_FA, data_Y, t_FA, K)
            
            if type(est) == mixture.GaussianMixture:
                    est_labels = est.predict(reduced_FA)
            else:
                est_labels = est.labels_
            
            new_data_X = np.c_[reduced_FA, est_labels]
            select_comp_supervised(new_data_X, data_Y, filename, K, est_name)
            #visualize(reduced_PCA, kmeansPCA, "K-means. PCA reduced", K)
            #visualize3D(sliced_ICA_data, est, est_name + " ICA reduced", K)
            #plot2D_mixture(reduced_ICA, 10, False)
            #plot_df_matrix(pd.DataFrame(reduced_RP))                

def test_RP(data_X, data_Y, filename, est_name, NUM_ATTR):
    for K in range(10, 11):
            # Initialize objects
            est = get_estimator(est_name, K)
            
            for i in range(3):
                # run Dimensionality Reduction Algorithms                        
                t_start = time()
                rp = random_projection.GaussianRandomProjection(n_components=NUM_ATTR)
                reduced_RP = rp.fit_transform(data_X)
                t_RP = time() - t_start
                # test
                #eblow(reduced_PCA, 20)
                test_estimator(filename, est, reduced_RP, data_Y, t_RP, K)
                
            if type(est) == mixture.GaussianMixture:
                    est_labels = est.predict(reduced_RP)
            else:
                est_labels = est.labels_
            
            new_data_X = np.c_[reduced_RP, est_labels]
            select_comp_supervised(new_data_X, data_Y, filename, K, est_name)
            #visualize(reduced_PCA, kmeansPCA, "K-means. PCA reduced", K)
            #visualize3D(sliced_ICA_data, est, est_name + " ICA reduced", K)
            #plot2D_mixture(reduced_ICA, 10, False)
            #plot_df_matrix(pd.DataFrame(reduced_RP))


def test_PCA(data_X, data_Y, filename, est_name, NUM_ATTR):
    for K in range(10, 11):
            # Initialize objects
            est = get_estimator(est_name, K)
            
            # run Dimensionality Reduction Algorithms                        
            t_start = time()
            reduced_PCA = PCA(n_components=NUM_ATTR).fit_transform(data_X)
            t_PCA = time() - t_start
            #reduced_ICA = find_ICA_comp(reduced_PCA, 5)
            #select_comp_supervised(reduced_RP, data_Y, filename, NUM_ATTR, est_name)
            # test
            #eblow(reduced_PCA, 20)
            test_estimator(filename, est, reduced_PCA, data_Y, t_PCA, K)
            
            if type(est) == mixture.GaussianMixture:
                    est_labels = est.predict(reduced_PCA)
            else:
                est_labels = est.labels_
            
            new_data_X = np.c_[reduced_PCA, est_labels]
            select_comp_supervised(new_data_X, data_Y, filename, K, est_name)
            #visualize(reduced_PCA, kmeansPCA, "K-means. PCA reduced", K)
            #visualize3D(sliced_ICA_data, est, est_name + " ICA reduced", K)
            #plot2D_mixture(reduced_ICA, 10, False)
            #plot_df_matrix(pd.DataFrame(reduced_RP))

            
def test_ICA(data_X, data_Y, filename, est_name, num_comp_keep):
    # get all reduced components    
    reduced_data, kurt_list, ica = find_ICA_comp(data_X)
    time_red = 0.4
    for K in [10]:
        #for num_comp_keep in range(2, data_X.shape[1]+1):
            # Initialize k-means objects
            est = get_estimator(est_name, K)
            
            # get all negative and num_comp_keep number of positive components
            sliced_ICA_data, sliced_kurt = slice_ICA_highest(reduced_data, kurt_list, num_comp_keep)
            #sliced_ICA_data, sliced_kurt = slice_ICA_data(reduced_data, kurt_list, num_comp_keep)
            #sliced_ICA_data, sliced_kurt = slice_ICA_list(reduced_data, kurt_list, range(12,23))
            #eblow(sliced_ICA_data, 20)            
            test_estimator(filename, est, sliced_ICA_data, data_Y, time_red, K)
            
            #test_estimator(filename, est, data_X, data_Y, 0, K)
            if type(est) == mixture.GaussianMixture:
                    est_labels = est.predict(sliced_ICA_data)
            else:
                est_labels = est.labels_
            
            new_data_X = np.c_[sliced_ICA_data,est_labels]
            select_comp_supervised(new_data_X, data_Y, filename, K, est_name)
            
            #visualize(sliced_ICA_data, est, est_name + " ICA reduced", K)
            #visualize3D(sliced_ICA_data, est, est_name + " ICA reduced", K)
            

def get_xy(df):
    X_raw = np.array(df.drop(df.columns[-1], 1))
    Y = np.array(df[df.columns[-1]])
    X = scale(X_raw.astype('float'))
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
    PC_num = 23
    IC_num = 23    
    for PC_num in range(1, 23):
        IC_num = PC_num
        
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
  
        
def write_to_file(row, filename):
    header = "Alg,NUM_ATTR,CV,Train,AUC,Time".split(",")
    is_file = os.path.exists(filename)                            
    with open(filename, 'ab') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not is_file:      
            writer.writerow(header)
        writer.writerows(row)
        
def select_comp_supervised(data_X, data_Y, filename, NUM_ATTR, est_name):
    save_to = "{0}{1}.csv".format(ann_folder, filename)
    
    # run supervised learning algorithm
    #if "ann" in filename:    
    cv_accuracy, train_score, auc, time_ann = fit_ann(data_X, data_Y)    
    #else:
    #cv_accuracy =fit_svm(data_X, data_Y) 
    
    print "\n{4}  train {0:.2f}   cv: {1:.2f}  auc: {2:.2f}   time: {3:.4f}".format(
                                   train_score, cv_accuracy, auc, time_ann, est_name)
    row = np.asarray([[est_name, NUM_ATTR, cv_accuracy, train_score, auc, time_ann]])
    write_to_file(row, save_to)
    
    
# Function used to print cross-validation scores
def training_score(est, X, y, cv):
    acc = cross_val_score(est, X, y, cv = cv, scoring='accuracy')
    if len(np.unique(y)) > 2:
        roc = 0
    else:
        roc = cross_val_score(est, X, y, cv = cv, scoring='roc_auc')
    #print '5-fold Train CV | Accuracy:', round(np.mean(acc), 3),'+/-', \
    #3round(np.std(acc), 3),'| ROC AUC:', round(np.mean(roc), 3), '+/-', round(np.std(roc), 3)
    acc = round(np.mean(acc), 4)*100
    roc = round(np.mean(roc), 4)*100
    return acc, roc

#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
def fit_ann(data_X, data_Y):
    ann = MLPClassifier(alpha=1, hidden_layer_sizes=(12, 12, 12), solver='adam',#sgd  adam
                        learning_rate='adaptive', learning_rate_init=0.001,#0.001, invscaling adaptive
                        momentum=0.4, max_iter=500)
    cv_score = 0

    t_start = time()                    
    ann.fit(data_X, data_Y)
    time_ann = time() - t_start
    
    cv = cross_validation.StratifiedShuffleSplit(data_Y, n_iter=5,test_size=0.3, random_state=42)
    
    train_score = round(ann.score(data_X, data_Y), 4)*100
    cv_score, auc = training_score(ann, data_X, data_Y, cv)
    
    print "\ntrain {0:.2f}   cv: {1:.2f}  auc: {2:.2f}   time {3:.4f}".format(train_score, cv_score, auc, time_ann)
    
    # run learning curve    
    #run_learning_curve(ann, data_X, data_Y, cv)
    return cv_score, train_score, auc, time_ann

def run_learning_curve(est, train_X, train_Y, cv):
    
    #train_sizes = [100, 500, 1000, 5000, 10000, 14700] - wine dataset
    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2399] # credit dataset
    title = 'Learning Curve (ANN, M={0}, L={1})'.format(0.2, 0.4)
    plt, train_scores, test_scores = plot_learning_curve(est, title, train_X, train_Y, 
                                                (0.2, 1.05), cv=cv, n_jobs=-1, 
                                                train_sizes = train_sizes)
    print train_scores, test_scores
    plt.show()

# http://stackoverflow.com/questions/23254700/inversing-pca-transform-with-sklearn-with-whiten-true
# https://roshansanthosh.wordpress.com/2015/06/21/pca-implementation-in-python-and-r/
def reconstruct_data(est, data_X):
    reconstructed_X = est.inverse_transform(data_X)
    return reconstructed_X


def get_reconstr_error(est, data_X_reduced, data_X_true):
    reconstructed_X = reconstruct_data(est, data_X_reduced)
    distance = np.linalg.norm(data_X_true-reconstructed_X,'fro')
    return distance


def reconstruct_PCA(data_X):
    dist = []
    for comp_num in range(1, 23):
        #selected_ICA, kurtosis_list, est = find_ICA_comp(credit_X)
        #selected_ICA, kurtosis_list = slice_ICA_data(selected_ICA, kurtosis_list, comp_num)
        est = PCA(n_components=comp_num)
        reduced_PCA = est.fit_transform(data_X)
        dist.append(get_reconstr_error(est, reduced_PCA, data_X))
    print dist


def test_ANN_PCA(data_X, data_Y, filename, est_name, NUM_ATTR=15):
    for NUM_ATTR in [6, 11]: #range(11, data_X.shape[1]+1):
        reduced_PCA = PCA(n_components=NUM_ATTR).fit_transform(data_X)
        select_comp_supervised(reduced_PCA, data_Y, filename, NUM_ATTR, est_name)
        
def test_ANN_ICA(data_X, data_Y, filename, est_name, NUM_ATTR=11):
    reduced_data, kurt_list, ica = find_ICA_comp(data_X)    
    #for NUM_ATTR in [15, 23]: #range(11, data_X.shape[1]+1):
    sliced_ICA_data, sliced_kurt = slice_ICA_list(reduced_data, kurt_list, range(6,11))
    select_comp_supervised(sliced_ICA_data, data_Y, filename, sliced_ICA_data.shape[1], est_name)
    select_comp_supervised(reduced_data, data_Y, filename, reduced_data.shape[1], est_name)
        
def test_ANN_RP(data_X, data_Y, filename, est_name, NUM_ATTR=15):
    for NUM_ATTR in [6, 11]: #range(11, data_X.shape[1]+1):
        for i in range(5):        
            rp = random_projection.GaussianRandomProjection(n_components=NUM_ATTR)
            reduced_RP = rp.fit_transform(data_X)
            select_comp_supervised(reduced_RP, data_Y, filename, NUM_ATTR, est_name)
        
def test_ANN_FA(data_X, data_Y, filename, est_name, NUM_ATTR=15):
    for NUM_ATTR in [6, 11]: #range(11, data_X.shape[1]+1):
        fa = FactorAnalysis(n_components=NUM_ATTR, random_state=42, max_iter=3000)
        reduced_FA = fa.fit_transform(data_X)
        select_comp_supervised(reduced_FA, data_Y, filename, NUM_ATTR, est_name)

def test_plain(data_X, data_Y, filename, est_name):  
    #for K in range(20, 30):
        K = 2
        est = get_estimator(est_name, K) 
        est.fit(data_X)
                                           
        test_estimator(filename, est, data_X, data_Y, 0, K)
        
        if type(est) == mixture.GaussianMixture:
                est_labels = est.predict(data_X)
        else:
            est_labels = est.labels_
        
        #test_estimator(filename, est, sliced_ICA_data, data_Y, time_red, K)
        #visualize(data_X, est, est_name + "Unreduced", K)
        #visualize3D(sliced_ICA_data, est, est_name + " ICA reduced", K)
        new_data_X = np.c_[data_X,est_labels]
        select_comp_supervised(new_data_X, data_Y, filename, K, est_name)
