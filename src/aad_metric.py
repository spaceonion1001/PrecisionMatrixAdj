import numpy as np
from sklearn.utils.extmath import fast_logdet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import normalize
from scipy.spatial.distance import mahalanobis
from scipy.stats import chisquare, chi2
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances
from sklearn.covariance import GraphicalLassoCV, LedoitWolf, GraphicalLasso
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from numpy.linalg import inv
import umap
from mpl_toolkits.mplot3d import Axes3D

from numba import njit
from numba_progress import ProgressBar
from statsmodels.stats.moment_helpers import cov2corr

from utils import *
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()
#from autoencoder import AE, train_model
#from torch.utils.data import TensorDataset, DataLoader
#import torch
np.random.seed(42)

from gmm import sim_gmm
from ocsvm import OCSVMBoost, NaiveBoostedOneClassSVM, OCSVMCVXPrimal, OCSVMCVXDual, OCSVMCVXPrimalRad, OCSVMCVXDualRad, ocsvm_solver, compute_rho, OCSVMRadAlt, SemiSupervisedOneClassSVM, OCSVMCVXPrimalMinimization, OCSVMMix
from aad_metric_model import BoostMetric
from aad_metric_model_v2 import AADMetricModel
from sklearn.preprocessing import Normalizer

from tsne import visualize_2d_embedding


import faulthandler
faulthandler.enable()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='C:\\Users\\xdink\\Documents\\Research\\OnlineMetricLearning\\data')
    parser.add_argument('--data', type=str, default='census')
    parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--v', type=float, default=1e-7)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--numfeats', type=int, default=10)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--simple_anom', action='store_true')
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--closer', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--negate', action='store_true')
    parser.add_argument('--original_mu', action='store_true')
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--use_top_k', action='store_true')
    parser.add_argument('--true_feedback', action='store_true')
    parser.add_argument('--full_k', action='store_true')
    parser.add_argument('--eigval_deweight', action='store_true')
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--use_oml', action='store_true')
    parser.add_argument('--same_class_dist', type=float, default=0.0)
    parser.add_argument('--anom_pairwise', action='store_true')
    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--save_suffix', type=str, default=None)
    parser.add_argument('--conserv_thresh', type=float, default=50)
    parser.add_argument('--w_upper', type=float, default=1.0)
    parser.add_argument('--oml_always', action='store_true')
    parser.add_argument('--opo', action='store_true')
    parser.add_argument('--prime_anomalies', action='store_true')
    parser.add_argument('--prime_nominals', action='store_true')
    parser.add_argument('--prime_limit', type=int, default=10)
    parser.add_argument('--use_mahal', action='store_true')
    parser.add_argument('--nom_deweight', action='store_true')
    parser.add_argument('--selection_thresh', type=int, default=99)
    parser.add_argument('--v2', action='store_true')

    args = parser.parse_args()
    if not args.closer:
        args.closer = False
    return args

def resolve_data(args):
    if args.data == 'census':
        return load_census_data(args)
    elif args.data == 'unsw':
        return load_unsw_data(args)
    elif args.data == 'kdd_donors':
        return load_kdd_donors_data(args)
    elif args.data == 'bank':
        return load_bank_data(args)
    elif args.data == 'campaign':
        return load_campaign_data(args)
    elif args.data == 'kddcup':
        return load_kddcup_data(args)
    elif args.data == 'wine':
        return load_wine_data(args)
    elif args.data == 'retinopathy':
        return load_retinopathy_data(args)
    elif args.data == 'cancer':
        return load_cancer_data(args)
    elif args.data == 'abalone':
        return load_abalone_data(args)
    elif args.data == 'thyroid':
        return load_thyroid_data(args)
    elif args.data == 'cardio':
        return load_cardio_data(args)
    elif args.data == 'mammography':
        return load_mammography_data(args)
    elif args.data == 'weather':
        return load_weather_data(args)
    elif args.data == 'yeast':
        return load_yeast_data(args)
    elif args.data == 'cifar':
        return load_cifar_data(args)
    elif args.data == 'fashion':
        return load_fashion_data(args)
    elif args.data == 'oxford':
        return load_oxford_data(args)
    elif args.data == 'nslkdd':
        return load_nslkdd_data(args)
    elif args.data == 'imagenet':
        return load_imagenet_data(args)
    elif args.data == 'msl':
        return load_msl_data(args)
    elif args.data == 'cifar_airplane':
        return load_cifar_airplane_data(args)
    elif args.data == 'cifar_bird':
        return load_cifar_bird_data(args)
    elif args.data == 'fashion_boot':
        return load_fashion_boot_data(args)
    elif args.data == 'fashion_sandal':
        return load_fashion_sandal_data(args)
    elif args.data == 'mnist':
        return load_mnist_data(args)
    elif args.data == 'cover':
        return load_cover_data(args)
    elif args.data == 'higgs':
        return load_higgs_data(args)
    elif args.data == 'spam':
        return load_spam_data(args)
    elif args.data == 'svhn':
        return load_svhn_data(args)
    else:
        print("Incorrect Dataset...")
        exit(1)


def init_covar(data, normalize=False, identity=False):
    A_t = np.cov(data.T)
    #A_t = np.corrcoef(data.T)
    A_t = A_t + 1e-5*np.eye(A_t.shape[0])

    if normalize:
        A_t = cov2corr(A_t)
    if identity:
        A_t = np.eye(A_t.shape[0])
    assert is_pos_def(A_t)
    return inv(A_t)


def mahalanobis_distance_matrix_vectorized(features, X):
    """
    Vectorized computation of the Mahalanobis distance matrix.

    Parameters:
    - X: An (N, d) matrix where each row is a data point.
    - S_inv: The precomputed inverse covariance matrix (d, d).

    Returns:
    - D_M: An (N, N) Mahalanobis distance matrix.
    """
    # Compute all pairwise differences
    delta = features[:, None, :] - features[None, :, :]  # Shape (N, N, d)
    
    # Compute the quadratic form for each pair
    dist_sq = np.einsum('ijk,kl,ijl->ij', delta, X, delta)  # (N, N)
    
    # Take the square root to get distances
    return np.sqrt(dist_sq)

def predict_percentile(args, features, X, labels, percentile=95, dist='mahalanobis'):
    mu = np.mean(features, axis=0)
    dists = []
    for i in range(features.shape[0]):
        a_i = features[i, :]
        if dist == 'mahalanobis':
            curr_dist = mahalanobis(a_i, mu, X)
        elif dist == 'euclidean':
            curr_dist = np.linalg.norm(a_i - mu)
        else:
            raise ValueError("Invalid Distance Metric")
        dists.append(curr_dist)
    dists = np.array(dists)
    anom_dists = dists[labels == 0]
    nominal_dists = dists[labels == 1]
    threshold = np.percentile(dists, percentile)
    preds = (dists > threshold).astype(int)
    preds = 1-preds # zero is anomaly
    print("Threshold {} Dist {} Total {}".format(threshold, dist, (1-preds).sum()))
    sns.histplot(anom_dists, alpha=0.5, label='anomalies', kde=True)
    sns.histplot(nominal_dists, alpha=0.5, label='nominal', kde=True)
    plt.legend(loc='best')
    plt.xlabel('Distance to Mean')
    plt.tight_layout()
    plt.savefig('./figures/debugging_figs/dist_hist_{}_{}_{}'.format("FINAL", dist, args.data))
    plt.close()
    return preds

def predict_top_5_percent(args, features, X, labels):
    mu = np.mean(features, axis=0)
    dists = []
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = mahalanobis(a_i, mu, X)
        dists.append(curr_dist)
    dists = np.array(dists)
    # sort indices in descending order by distance
    sorted_indices = np.argsort(dists)[::-1]
    top_5_percent = int(0.05*features.shape[0])
    top_5_indices = sorted_indices[:top_5_percent]
    print("Size of Top 5 Percent {}".format(top_5_indices.shape))
    preds = np.ones(features.shape[0])
    preds[top_5_indices] = 0 # anomaly

    return preds


def fast_kernel(X1, X2):
    return pairwise_distances(X1, X2, metric=lambda x, y: np.dot(x, y))

def plot_precision_recall(args, bm):
    plt.plot(np.arange(args.iters), bm.f1s, label="F1")
    plt.plot(np.arange(args.iters), bm.precisions, label='Precision')
    plt.plot(np.arange(args.iters), bm.recalls, label='Recall')
    plt.xlabel("Iterations")
    plt.ylabel("Scores")
    plt.legend(loc='best')
    plt.title("Scores {}".format(args.data))
    plt.savefig('figures/debugging_figs/scores_{}'.format(args.data))
    plt.close()

    plt.plot(np.arange(args.iters), bm.num_preds)
    plt.xlabel("Iterations")
    plt.ylabel("Number of Predicted Anomalies (ChiSq)")
    #plt.legend(loc='best')
    plt.title("Number Predicted {}".format(args.data))
    plt.savefig('figures/debugging_figs/numpreds_{}'.format(args.data))
    plt.close()

def main(args):
    data, features, labels = resolve_data(args)

     # unique count of labels
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
    features = remove_bad_features(features)
    scaler = StandardScaler()
    
    # features = scaler.fit_transform(features)

    # clf = DecisionTreeClassifier(max_depth=2)
    # clf.fit(features, labels)
    # print("Score {}".format(clf.score(features, labels)))
    # print("Feature Importance {}".format(clf.feature_importances_))
    # print("Feature Importance Argmax {}".format(clf.feature_importances_.argmax()))

    total_data = features.copy()
    total_labels = labels.copy()

    if args.data != 'wine':
        #features, labels = smart_sampling(features=features, labels=labels, num_anoms=10, num_nominals=100)
        features = add_epsilon_noise(features=features)
        if args.data == 'bank':
            labels = 1-labels # switch 0 and 1
    else:
        labels[labels != 0] = 1

    features = scaler.fit_transform(features)
    # randomly select 10000 samples from features
    labels_svm = labels.copy()
    labels_svm[labels_svm == 0] = -1
    features_subset, labels_subset = shuffle(features, labels_svm, random_state=42)
    features_subset = features_subset[:10000]
    labels_subset = labels_subset[:10000]
   

    init_dist_mat = init_covar(features, normalize=args.normalize, identity=args.identity)
    print("Data {} Shape {}".format(args.data, features.shape))

    bm = BoostMetric(data=features, labels=labels, init_dist_mat=init_dist_mat, args=args, v=args.v, J=args.iters, top_k=args.k)
    X = bm.iterate()
    w, Z = bm.get_w_Z()
    us = bm.get_us()
    argmax_u = np.argmax(us, axis=0)
    print("Argmax U {}".format(us[argmax_u]))
    print("Argmax of U in W {}".format(w[argmax_u]))
    #print(us)
    print("Length of w {}".format(len(w)))
    plot_precision_recall(args, bm)
   
    #print("SVM Boost: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, svm_boost_preds), precision_score(1-labels, svm_boost_preds), recall_score(1-labels, svm_boost_preds)))
    #print("SVM Naive Boost: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, naive_preds), precision_score(1-labels, naive_preds), recall_score(1-labels, naive_preds)))
    print("BoostMetric: F1 {}, Precision {}, Recall {}".format(bm.f1s[-1], bm.precisions[-1], bm.recalls[-1]))
    #total_distances = mahal_all_points(features, X)
    #knn_clf = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    #knn_clf.fit(total_distances, labels)
    #knn_preds = knn_clf.predict(total_distances)
    #print("KNN: F1 {}, Precision {}, Recall {}".format(f1_score(labels, knn_preds), precision_score(labels, knn_preds), recall_score(labels, knn_preds)))
    #knn_clf_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    #knn_clf_euclidean.fit(features, labels)
    #knn_preds_euclidean = knn_clf_euclidean.predict(features)
    #print("KNN Euclidean: F1 {}, Precision {}, Recall {}".format(f1_score(labels, knn_preds_euclidean), precision_score(labels, knn_preds_euclidean), recall_score(labels, knn_preds_euclidean)))
    preds_mahal_percentile = predict_percentile(args, features, X, labels=labels, percentile=95, dist='mahalanobis')
    print("Mahal Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_percentile), precision_score(1-labels, 1-preds_mahal_percentile), recall_score(1-labels, 1-preds_mahal_percentile)))
    preds_euclidean_percentile = predict_percentile(args, features, X=None, labels=labels, percentile=95, dist='euclidean')
    print("Euclidean Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_euclidean_percentile), precision_score(1-labels, 1-preds_euclidean_percentile), recall_score(1-labels, 1-preds_euclidean_percentile)))
    pca_clf = PCA(n_components=2)
    pca_data = pca_clf.fit_transform(features)
    preds_pca_euclidean_percentile = predict_percentile(args, pca_data, X=None, labels=labels, percentile=95, dist='euclidean')
    print("PCA Euclidean Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_pca_euclidean_percentile), precision_score(1-labels, 1-preds_pca_euclidean_percentile), recall_score(1-labels, 1-preds_pca_euclidean_percentile)))
    preds_mahal_default_percentile = predict_percentile(args, features, X=inv(np.cov(features.T)), labels=labels, percentile=95, dist='mahalanobis')
    print("Mahal Default Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_default_percentile), precision_score(1-labels, 1-preds_mahal_default_percentile), recall_score(1-labels, 1-preds_mahal_default_percentile)))
    

    dists = []
    mu = np.mean(features, axis=0)
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = np.linalg.norm(a_i - mu)
        dists.append(curr_dist)
    dists = np.array(dists)
    anom_dists = dists[labels == 0]
    nominal_dists = dists[labels == 1]

    # save histogram of distances
    sns.histplot(anom_dists, alpha=0.5, label='anomalies', kde=True)
    sns.histplot(nominal_dists, alpha=0.5, label='nominal', kde=True)
    plt.legend(loc='best')
    plt.xlabel('Distance to Mean')
    plt.tight_layout()
    plt.savefig('./figures/debugging_figs/dist_hist_euclid_{}'.format(args.data))
    plt.close()

    ##########

    #final_classification(features=features, labels=labels, X=X)
    ##############################

def main_true(args):
    data, features, labels = resolve_data(args)

     # unique count of labels
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
    features = remove_bad_features(features)
    scaler = StandardScaler()
    
    # features = scaler.fit_transform(features)

    # clf = DecisionTreeClassifier(max_depth=2)
    # clf.fit(features, labels)
    # print("Score {}".format(clf.score(features, labels)))
    # print("Feature Importance {}".format(clf.feature_importances_))
    # print("Feature Importance Argmax {}".format(clf.feature_importances_.argmax()))

    total_data = features.copy()
    total_labels = labels.copy()
    binary_columns = [i for i in range(features.shape[1]) if np.unique(features[:, i]).size == 2]
    numeric_columns = [i for i in range(features.shape[1]) if i not in binary_columns]
    print("Number of Binary Columns {}".format(len(binary_columns)))
    #############
    # if len(binary_columns) > 0:
    #     features_lim = features[:, binary_columns]

    #     reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, metric='jaccard')

    #     # Fit and transform the data
    #     X_umap = reducer.fit_transform(features_lim)

    #     # Create a scatter plot
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(X_umap[labels == 0, 0], X_umap[labels == 0, 1], label="Class 0", alpha=0.6, c='blue', s=25)
    #     plt.scatter(X_umap[labels == 1, 0], X_umap[labels == 1, 1], label="Class 1", alpha=0.4, c='orange', s=10)
    #     mean_class0 = np.mean(X_umap[labels == 0], axis=0)
    #     mean_class1 = np.mean(X_umap[labels == 1], axis=0)
    #     plt.scatter(mean_class0[0], mean_class0[1], c='blue', edgecolors='black', s=300, marker='X', label="Class 0 Center")
    #     plt.scatter(mean_class1[0], mean_class1[1], c='orange', edgecolors='black', s=300, marker='X', label="Class 1 Center")
    #     plt.legend()
    #     plt.title("t-SNE Visualization of Two-Class Dataset")
    #     plt.xlabel("t-SNE Dimension 1")
    #     plt.ylabel("t-SNE Dimension 2")
    #     plt.savefig('./figures/debugging_figs_true/tsne_{}_binary'.format(args.data))
    #     plt.close()
    ###########
    
    if args.data != 'wine':
        #features, labels = smart_sampling(features=features, labels=labels, num_anoms=10, num_nominals=100)
        if args.data == 'bank':
            labels = 1-labels # switch 0 and 1
        if args.data in ['bank', 'unsw', 'nslkdd', 'campaign']:
            features = add_epsilon_noise(features=features) # numerical stability
        

    else:
        labels[labels != 0] = 1

    #features = scaler.fit_transform(features)
    ###########
    numeric_features_scaled = scaler.fit_transform(features[:, numeric_columns])
    binary_features = features[:, binary_columns]

    features_stacked = np.hstack((numeric_features_scaled, binary_features))
    features = features_stacked
    ###########




    ###########################################
    # # Normalize each row to unit L2 norm
    # l2_normalizer = Normalizer(norm='l2')
    # numeric_features_l2 = l2_normalizer.fit_transform(numeric_features_scaled)

    # # Stack normalized numeric features with binary features
    # features = np.hstack((numeric_features_l2, binary_features))
    # ###########################################




    ###########################################
    # # Apply PCA whitening (full dimensionality)
    # pca = PCA(whiten=True)
    # numeric_features_whitened = pca.fit_transform(numeric_features_scaled)

    # # Stack whitened numeric features with binary features
    # features = np.hstack((numeric_features_whitened, binary_features))
    ###########################################




    # shuffle the data
    features, labels = shuffle(features, labels, random_state=42)
    ###################
    # reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, metric='euclidean', min_dist=0.1)

    # # Fit and transform the data
    # X_umap = reducer.fit_transform(features)

    # # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_umap[labels == 0, 0], X_umap[labels == 0, 1], label="Class 0", alpha=0.6, c='blue', s=25)
    # plt.scatter(X_umap[labels == 1, 0], X_umap[labels == 1, 1], label="Class 1", alpha=0.4, c='orange', s=10)
    # mean_class0 = np.mean(X_umap[labels == 0], axis=0)
    # mean_class1 = np.mean(X_umap[labels == 1], axis=0)
    # plt.scatter(mean_class0[0], mean_class0[1], c='blue', edgecolors='black', s=300, marker='X', label="Class 0 Center")
    # plt.scatter(mean_class1[0], mean_class1[1], c='orange', edgecolors='black', s=300, marker='X', label="Class 1 Center")
    # plt.legend()
    # plt.title("t-SNE Visualization of Two-Class Dataset")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.savefig('./figures/debugging_figs_true/tsne_{}'.format(args.data))
    # plt.close()

    # # Initialize UMAP for 3D
    # reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, metric='euclidean', min_dist=0.1)
    # X_umap_3d = reducer_3d.fit_transform(features)

    # # 3D scatter plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_umap_3d[labels == 0, 0], X_umap_3d[labels == 0, 1], X_umap_3d[labels == 0, 2], label="Class 0", alpha=0.6, c='blue', s=25)
    # ax.scatter(X_umap_3d[labels == 1, 0], X_umap_3d[labels == 1, 1], X_umap_3d[labels == 1, 2], label="Class 1", alpha=0.4, c='orange', s=10)
    # ax.set_title("UMAP 3D Visualization of Two-Class Dataset")
    # ax.set_xlabel("UMAP Dimension 1")
    # ax.set_ylabel("UMAP Dimension 2")
    # ax.set_zlabel("UMAP Dimension 3")
    # ax.legend()
    # plt.savefig('./figures/debugging_figs_true/tsne_{}_3d'.format(args.data))
    # plt.close()

    # # Fit a linear classifier
    # clf = LogisticRegression()
    # clf.fit(features, labels)

    # # Check accuracy
    # y_pred = clf.predict(features)
    # print("\n--------------------------------------------------------------------------------------------")
    # print("CLF Accuracy {} F1 {}".format(accuracy_score(labels, y_pred), f1_score(labels, y_pred)))
    # print("Proportion of Nominals {}".format(labels.sum()/len(labels)))
    # print('--------------------------------------------------------------------------------------------')
    # print()

    # exit()
    ###################
    
    # randomly select 10000 samples from features
    # labels_svm = labels.copy()
    # labels_svm[labels_svm == 0] = -1
    # features_subset, labels_subset = shuffle(features, labels_svm, random_state=42)
    # features_subset = features_subset[:10000]
    # labels_subset = labels_subset[:10000]
   

    # visualize_2d_embedding(features, labels, method='umap', args=args)
    # visualize_2d_embedding(features, labels, method='tsne', args=args)

    ###
    # GMM STUFF
    # k = 5
    # print("Fitting GMM with {} components".format(k))
    # gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    # gmm.fit(features)
    # print("GMM Weights {}".format(gmm.weights_))
    # print("GMM Means {}".format(gmm.means_))
    # print("GMM Covariances {}".format(gmm.covariances_))
    # exit()
    for seed in range(42, 43):
        print(">>>> SEED {} <<<<".format(seed))
        print("Data {} Shape {}".format(args.data, features.shape))
        
        init_dist_mat = init_covar(features, normalize=args.normalize, identity=args.identity)
        

        bm = BoostMetric(data=features.copy(), labels=labels.copy(), init_dist_mat=init_dist_mat.copy(), args=args, v=args.v, J=args.iters, top_k=args.k, seed=seed)
        if args.v2:
            print("Using V2")
            bm = AADMetricModel(data=features.copy(), labels=labels.copy(), init_dist_mat=init_dist_mat.copy(), args=args, v=args.v, J=args.iters, top_k=args.k, seed=seed)
        X = bm.iterate()
        #w, Z = bm.get_w_Z()
        #us = bm.get_us()
        #argmax_u = np.argmax(us, axis=0)
        #print("Argmax U {}".format(us[argmax_u]))
        #print("Argmax of U in W {}".format(w[argmax_u]))
        #print(us)
        #print("Length of w {}".format(len(w)))
        #print("Length of Z {}".format(len(Z)))
        #plot_precision_recall(args, bm)
    
        #print("BoostMetric: F1 {}, Precision {}, Recall {}".format(bm.f1s[-1], bm.precisions[-1], bm.recalls[-1]))
        
        preds_mahal_percentile = predict_percentile(args, features, X, labels=labels, percentile=95, dist='mahalanobis')
        print("Mahal Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_percentile), precision_score(1-labels, 1-preds_mahal_percentile), recall_score(1-labels, 1-preds_mahal_percentile)))
        preds_euclidean_percentile = predict_percentile(args, features, X=None, labels=labels, percentile=95, dist='euclidean')
        print("Euclidean Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_euclidean_percentile), precision_score(1-labels, 1-preds_euclidean_percentile), recall_score(1-labels, 1-preds_euclidean_percentile)))
        pca_clf = PCA(n_components=2)
        pca_data = pca_clf.fit_transform(features)
        preds_pca_euclidean_percentile = predict_percentile(args, pca_data, X=None, labels=labels, percentile=95, dist='euclidean')
        print("PCA Euclidean Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_pca_euclidean_percentile), precision_score(1-labels, 1-preds_pca_euclidean_percentile), recall_score(1-labels, 1-preds_pca_euclidean_percentile)))
        preds_mahal_default_percentile = predict_percentile(args, features, X=inv(np.cov(features.T)), labels=labels, percentile=95, dist='mahalanobis')
        print("Mahal Default Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_default_percentile), precision_score(1-labels, 1-preds_mahal_default_percentile), recall_score(1-labels, 1-preds_mahal_default_percentile)))
        top_5_percent_preds = predict_top_5_percent(args, features, X, labels)
        print("Top 5 Percent: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-top_5_percent_preds), precision_score(1-labels, 1-top_5_percent_preds), recall_score(1-labels, 1-top_5_percent_preds)))
        # save the precision score to a file specific to the data
        with open('./results/precision_scores_{}_{}.txt'.format(args.data, args.save_suffix), 'a') as f:
            f.write("Seed {} Precision {}\n".format(seed, precision_score(1-labels, 1-preds_mahal_percentile)))
            # add recall and f1
            f.write("Seed {} Recall {}\n".format(seed, recall_score(1-labels, 1-preds_mahal_percentile)))
            f.write("Seed {} F1 {}\n".format(seed, f1_score(1-labels, 1-preds_mahal_percentile)))

        
        print(">>>>__________<<<<")
        del bm



if __name__ == "__main__":
    args = parse_arguments()
    #main(args)
    main_true(args)
    #main_boost(args)

   