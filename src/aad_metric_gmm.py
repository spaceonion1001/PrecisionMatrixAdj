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
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances, silhouette_score, roc_auc_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from numpy.linalg import inv
import umap
from mpl_toolkits.mplot3d import Axes3D
import time

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
#from aad_metric_model_v2 import AADMetricModel
from aad_metric_model_gmm import AADMetricModel
from sklearn.preprocessing import Normalizer

from tsne import visualize_2d_embedding


import faulthandler
faulthandler.enable()

from copy import deepcopy

import gapstatistics
from gapstatistics.gapstatistics import GapStatistics

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
    parser.add_argument('--interleaving', action='store_true')

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
    elif 'cifar' in args.data:
        return load_cifar_numbered_data(args, num=int(args.data.split('_')[-1]))
    elif 'fashion' in args.data:
        return load_fashion_numbered_data(args, num=int(args.data.split('_')[-1]))
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

def predict_top_5_percent_gmms(features, gmms):
    k = len(gmms)
    # find the distance for each point to the mean of each GMM
    # use the Mahalanobis distance
    dists = []
    for i in range(k):
        gmm = gmms[i]
        mean_vec = gmm.mu
        A_t = gmm.curr_dist_mat
        dists.append(calc_anomaly_scores(features, A_t, mean_vec))
    dists = np.array(dists)
    # find each point's minimum distance to the GMM means
    min_dists = np.min(dists, axis=0)
    dists = min_dists
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



def get_dists(features, gmms, dists, prev_selected_gmm):
    k = len(gmms)
    # find the distance for each point to the mean of each GMM
    # use the Mahalanobis distance
    gmm = gmms[prev_selected_gmm]
    mean_vec = gmm.mu
    A_t = gmm.curr_dist_mat
    dists[prev_selected_gmm, :] = calc_anomaly_scores(features, A_t, mean_vec)
    
    return dists

def get_query_point(features, labels, masked_features, dists):
    use_data = features[masked_features, :]
    # print("Use Data Shape: ", use_data.shape)
    use_labels = labels[masked_features]
    use_dists = dists[:, masked_features]

    min_dists = np.min(use_dists, axis=0)
    max_min_dist = np.max(min_dists)
    selected_idx = np.argmax(min_dists)
    # print("Selected Index {} Max Min Dist {}".format(selected_idx, max_min_dist))

    # Fix here: convert to original index
    selected_indices = np.flatnonzero(masked_features)
    original_idx = selected_indices[selected_idx]

    selected_gmm = np.argmin(dists[:, original_idx])
    # print("Selected GMM {}".format(selected_gmm))
    selected_point = use_data[selected_idx, :]
    selected_label = use_labels[selected_idx]

    return selected_point, selected_label, selected_idx, selected_gmm

def get_query_point_interleaving(features, labels, masked_features, selected_gmm, dists):
    # interleaving gets the farthest point from any GMM, regardless of distance from others

    use_data = features[masked_features, :]
    # print("Use Data Shape: ", use_data.shape)
    use_labels = labels[masked_features]
    use_dists = dists[selected_gmm, masked_features]

    # min_dists = np.min(use_dists, axis=0)
    # max_min_dist = np.max(min_dists)
    selected_idx = np.argmax(use_dists)
    # print("Selected Index {} Max Min Dist {}".format(selected_idx, max_min_dist))

    # Fix here: convert to original index
    selected_indices = np.flatnonzero(masked_features)
    original_idx = selected_indices[selected_idx]

    #selected_gmm = np.argmin(dists[:, original_idx])
    # print("Selected GMM {}".format(selected_gmm))
    selected_point = use_data[selected_idx, :]
    selected_label = use_labels[selected_idx]

    return selected_point, selected_label, selected_idx

# def get_query_point(features, labels, masked_features, gmms):
#     use_data = features[masked_features, :]
#     print("Use Data Shape: ", use_data.shape)
#     use_labels = labels[masked_features]
#     k = len(gmms)
#     # find the distance for each point to the mean of each GMM
#     # use the Mahalanobis distance
#     dists = []
#     for i in range(k):
#         gmm = gmms[i]
#         mean_vec = gmm.mu
#         A_t = gmm.curr_dist_mat
#         dists.append(calc_anomaly_scores(use_data, A_t, mean_vec))
#     dists = np.array(dists)
#     # find each point's minimum distance to the GMM means
#     min_dists = np.min(dists, axis=0)
#     # find the point with the maximum minimum distance
#     max_min_dist = np.max(min_dists)
#     selected_idx = np.argmax(min_dists)
#     print("Selected Index {} Max Min Dist {}".format(selected_idx, max_min_dist))
#     selected_gmm = np.argmin(dists[:, selected_idx])
#     print("Selected GMM {}".format(selected_gmm))
#     selected_point = use_data[selected_idx, :]
#     selected_label = use_labels[selected_idx]


#     return selected_point, selected_label, selected_idx, selected_gmm

def get_anomaly_scores(features, gmms):
    dists = []
    k = len(gmms)
    for i in range(k):
        gmm = gmms[i]
        mean_vec = gmm.mu
        A_t = gmm.curr_dist_mat
        dists.append(calc_anomaly_scores(features, A_t, mean_vec))
    dists = np.array(dists)
    # find each point's minimum distance to the GMM means
    min_dists = np.min(dists, axis=0)

    return min_dists

def get_Z_update_mask(gmm, selected_point, selected_idx, masked_features):
    mean_vec = gmm.mu
    diff = selected_point - mean_vec
    u_vec = diff/np.linalg.norm(diff)
    z_vec = u_vec
    z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
    z_vec = np.expand_dims(z_vec, 1)
    curr_Z = z_vec@z_vec.T

    selected_indices = np.flatnonzero(masked_features)
    original_idx = selected_indices[selected_idx]
    # print("Removing Point {}".format(original_idx))
    
    masked_features[original_idx] = False

    return curr_Z, masked_features, original_idx

def update_gmm(gmm, curr_Z, selected_point, selected_label):
    is_anomaly = (selected_label == 0)
    gmm.single_iterate(curr_Z, is_anomaly, selected_point)
    

def run_aad(args, features, labels, gmms):
    masked_features = np.ones(features.shape[0], dtype=bool)
    # calculate the initial distances
    dists = np.zeros((len(gmms), features.shape[0]))
    # find the distance for each point to the mean of each GMM
    # use the Mahalanobis distance
    for i in range(len(gmms)):
        gmm = gmms[i]
        mean_vec = gmm.mu
        A_t = gmm.curr_dist_mat
        dists[i, :] = calc_anomaly_scores(features, A_t, mean_vec)
    queried_indices = []
    num_anomalies_queried = 0
    num_anomalies_queried_list = []
    for b in range(args.iters):
        # print("Budget {}".format(b))
        # get the query point
        query_point, query_label, query_idx, selected_gmm = get_query_point(features, labels, masked_features, dists=dists)
        # print("Query Point {} Label {}".format(query_idx, query_label))
        if query_label == 0:
            # print("Anomaly Found")
            num_anomalies_queried += 1
        num_anomalies_queried_list.append(num_anomalies_queried)
        curr_Z, masked_features, original_idx = get_Z_update_mask(gmms[selected_gmm], query_point, query_idx, masked_features)
        queried_indices.append(original_idx)
        update_gmm(gmms[selected_gmm], curr_Z, query_point, query_label)
        # update the distances
        dists = get_dists(features, gmms, dists, selected_gmm)
    
    return gmms, num_anomalies_queried, num_anomalies_queried_list, queried_indices

def run_aad_interleaving(args, features, labels, gmms):
    masked_features = np.ones(features.shape[0], dtype=bool)
    # calculate the initial distances
    dists = np.zeros((len(gmms), features.shape[0]))
    # find the distance for each point to the mean of each GMM
    # use the Mahalanobis distance
    for i in range(len(gmms)):
        gmm = gmms[i]
        mean_vec = gmm.mu
        A_t = gmm.curr_dist_mat
        dists[i, :] = calc_anomaly_scores(features, A_t, mean_vec)
    queried_indices = []
    num_anomalies_queried = 0
    num_anomalies_queried_list = []
    selected_gmm = 0
    for b in range(args.iters):
        # print("Budget {}".format(b))
        # get the query point
        #query_point, query_label, query_idx, selected_gmm = get_query_point(features, labels, masked_features, dists=dists)
        query_point, query_label, query_idx = get_query_point_interleaving(features, labels, masked_features, selected_gmm, dists=dists)
        # print("Query Point {} Label {}".format(query_idx, query_label))
        if query_label == 0:
            # print("Anomaly Found")
            num_anomalies_queried += 1
        num_anomalies_queried_list.append(num_anomalies_queried)
        curr_Z, masked_features, original_idx = get_Z_update_mask(gmms[selected_gmm], query_point, query_idx, masked_features)
        queried_indices.append(original_idx)
        update_gmm(gmms[selected_gmm], curr_Z, query_point, query_label)
        # update the distances
        dists = get_dists(features, gmms, dists, selected_gmm)
        selected_gmm += 1
        if selected_gmm >= len(gmms):
            selected_gmm = 0
    
    return gmms, num_anomalies_queried, num_anomalies_queried_list, queried_indices

# def run_aad(args, features, labels, gmms):
#     masked_features = np.ones(features.shape[0], dtype=bool)
#     queried_indices = []
#     num_anomalies_queried = 0
#     num_anomalies_queried_list = []
#     for b in range(args.iters):
#         print("Budget {}".format(b))
#         # get the query point
#         query_point, query_label, query_idx, selected_gmm = get_query_point(features, labels, masked_features, gmms)
#         print("Query Point {} Label {}".format(query_idx, query_label))
#         if query_label == 0:
#             print("Anomaly Found")
#             num_anomalies_queried += 1
#         else:
#             print("Nominal Found")
#         num_anomalies_queried_list.append(num_anomalies_queried)
#         curr_Z, masked_features, original_idx = get_Z_update_mask(gmms[selected_gmm], query_point, query_idx, masked_features)
#         queried_indices.append(original_idx)
#         update_gmm(gmms[selected_gmm], curr_Z, query_point, query_label)
    
#     return gmms, num_anomalies_queried, num_anomalies_queried_list, queried_indices

@njit
def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis_fast_uv(v, mean_vec, A_t)
        dists.append(curr_dist)

    return np.array(dists)


def main_true(args):
    data, features, labels = resolve_data(args)

     # unique count of labels
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
    features = remove_bad_features(features)
    scaler = StandardScaler()
    

    total_data = features.copy()
    total_labels = labels.copy()
    binary_columns = [i for i in range(features.shape[1]) if np.unique(features[:, i]).size == 2]
    numeric_columns = [i for i in range(features.shape[1]) if i not in binary_columns]
    print("Number of Binary Columns {}".format(len(binary_columns)))
    
    if args.data != 'wine':
        #features, labels = smart_sampling(features=features, labels=labels, num_anoms=10, num_nominals=100)
        if args.data == 'bank':
            labels = 1-labels # switch 0 and 1
        if args.data in ['bank', 'unsw', 'nslkdd', 'campaign']:
            features_numeric = features[:, numeric_columns]
            features_binary = features[:, binary_columns]

            # Add epsilon noise only to binary features
            #epsilon = 1e-5
            #features_binary_noisy = features_binary + epsilon * np.random.randn(*features_binary.shape)
            features_binary_noisy = add_epsilon_noise(features_binary)

            # Reconstruct the full feature matrix
            features = np.zeros_like(features)
            features[:, numeric_columns] = features_numeric
            features[:, binary_columns] = features_binary_noisy
        

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

    # # ##
    # # # GMM STUFF
    # k_range = range(2, 11)
    # avg_bics = []
    # for trial_k in tqdm(k_range):
    #     bics = []
    #     for seed in range(30, 32):
    #         init_precisions = np.array([np.ones(20) for _ in range(trial_k)])
    #         gmm = GaussianMixture(n_components=trial_k, covariance_type='diag', random_state=seed, precisions_init=init_precisions, n_init=5)
    #         # gmm = GaussianMixture(n_components=trial_k, covariance_type='full', random_state=seed, n_init=5)
    #         pca = PCA(n_components=20, random_state=seed)
    #         features_reduced = pca.fit_transform(features)
    #         gmm.fit(features_reduced)
    #         bics.append(gmm.bic(features_reduced))
    #     avg_bics.append(np.mean(bics))
    # plot_path = f"./results_gmms/{args.data}_gmm_elbow.png"

    # plt.figure(figsize=(6, 4))
    # plt.plot(k_range, avg_bics, marker='o')
    # plt.xlabel("Number of GMM Components (k)")
    # plt.ylabel("Average BIC over seeds")
    # plt.title(f"GMM Elbow Plot ({args.data})")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(plot_path)
    # plt.close()
    best_k, metrics = evaluate_k_range(features, k_range=range(2, 11), max_pca_dim=20, random_state=42, plot_path=f"./results_kmeans_reduced/{args.data}_gmm_elbow.png")
    # best_k, metrics = evaluate_k_range(features, k_range=range(2, 11), max_pca_dim=20, random_state=42, plot_path=None)
    # evaluate_kmeans(features_reduced, k_range=range(2, 11), save_path="./results_kmeans_reduced", dataset_name=args.data)
    runtimes = []
    for seed in range(40, 50):
        print(">>>> SEED {} <<<<".format(seed))
        print("Data {} Shape {}".format(args.data, features.shape))
        full_covariance = False
        k = best_k
        # k = 7
        print("Fitting GMM with {} components".format(k))
        start_time = time.time()
        if full_covariance:
            # init_precisions = np.array([np.eye(features.shape[1]) for _ in range(k)])
            init_precisions = np.tile(np.eye(features.shape[1])[None, :, :], (k, 1, 1))
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=seed, precisions_init=init_precisions, n_init=5, init_params='kmeans')
            # gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=seed, n_init=5)
        else:
            init_precisions = np.array([np.ones(features.shape[1]) for _ in range(k)])
            gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=seed, precisions_init=init_precisions, n_init=5, init_params='kmeans')
            # init_precisions = np.array([np.ones(features.shape[1]) for _ in range(k)])
            # # init_precisions = np.ones((k, features.shape[1]), dtype=np.float64)
            # gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=seed, precisions_init=init_precisions, n_init=5, max_iter=100)
            # gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=seed, n_init=5)

        gmm.fit(features)
        if gmm.converged_:
            print("GMM converged.")
            print(f"Converged in {gmm.n_iter_} EM iterations")
        else:
            print("GMM did not converge.")
        gmms = []
        gmms_init = []
        for component in range(k):
            component_mean = gmm.means_[component]
            component_cov = gmm.covariances_[component]
            component_precision = gmm.precisions_[component]
            component_weight = gmm.weights_[component]
            if full_covariance:
                curr_precision = component_precision
            else:
                curr_precision = np.diag(component_precision)
            bm_k = AADMetricModel(data=features.copy(), labels=labels.copy(), init_dist_mat=curr_precision.copy(), mu=component_mean, args=args, v=args.v, J=args.iters, top_k=args.k, seed=seed)
            gmms.append(bm_k)
            gmms_init.append(deepcopy(bm_k))
        
        
        if args.interleaving:
            print("> Running Interleaving AAD <")
            gmms, num_anomalies_queried, num_anomalies_queried_list, selected_indices = run_aad_interleaving(args, features, labels, gmms)
        else:
            print("> Running AAD <")
            gmms, num_anomalies_queried, num_anomalies_queried_list, selected_indices = run_aad(args, features, labels, gmms)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        print("Selected indices {}".format(selected_indices))
        print("Num Anomalies Queried {}".format(num_anomalies_queried))

        preds_mahal_default_percentile = predict_percentile(args, features, X=inv(np.cov(features.T)), labels=labels, percentile=95, dist='mahalanobis')
        print("Mahal Default Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_default_percentile), precision_score(1-labels, 1-preds_mahal_default_percentile), recall_score(1-labels, 1-preds_mahal_default_percentile)))
        preds_mahal_gmm_percentile = predict_top_5_percent_gmms(features, gmms_init)
        print("Mahal GMM Default Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_gmm_percentile), precision_score(1-labels, 1-preds_mahal_gmm_percentile), recall_score(1-labels, 1-preds_mahal_gmm_percentile)))
        top_5_percent_preds = predict_top_5_percent_gmms(features, gmms)
        print("Top 5 Percent: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-top_5_percent_preds), precision_score(1-labels, 1-top_5_percent_preds), recall_score(1-labels, 1-top_5_percent_preds)))
        anomaly_scores = get_anomaly_scores(features, gmms)
        anomaly_scores_init = get_anomaly_scores(features, gmms_init)
        print("ROC AUC GMM Default {}: {}".format(args.data, roc_auc_score(1-labels, anomaly_scores_init)))
        print("ROC AUC GMM Mahal {}: {}".format(args.data, roc_auc_score(1-labels, anomaly_scores)))
        print("Avg Prec Score GMM Default {}: {}".format(args.data, average_precision_score(1-labels, anomaly_scores_init)))
        print("Avg Prec Score GMM Mahal {}: {}".format(args.data, average_precision_score(1-labels, anomaly_scores)))
        # save the precision score to a file specific to the data
        if args.interleaving:
            args.save_suffix = "interleaving"
        if full_covariance:
            os.makedirs('./results_gmms_seeds_full', exist_ok=True)
            with open('./results_gmms_seeds_full/precision_scores_{}_{}_{}.txt'.format(args.data, args.save_suffix, best_k), 'a') as f:
                f.write("Seed {} Precision {}\n".format(seed, precision_score(1-labels, 1-top_5_percent_preds)))
                # add recall and f1
                f.write("Seed {} Recall {}\n".format(seed, recall_score(1-labels, 1-top_5_percent_preds)))
                f.write("Seed {} F1 {}\n".format(seed, f1_score(1-labels, 1-top_5_percent_preds)))
                f.write("Seed {} ROC AUC {}\n".format(seed, roc_auc_score(1-labels, anomaly_scores)))
                f.write("Seed {} Avg Prec {}\n".format(seed, average_precision_score(1-labels, anomaly_scores)))
        else:
            with open('./results_gmms_seeds/precision_scores_{}_{}_{}.txt'.format(args.data, args.save_suffix, best_k), 'a') as f:
                f.write("Seed {} Precision {}\n".format(seed, precision_score(1-labels, 1-top_5_percent_preds)))
                # add recall and f1
                f.write("Seed {} Recall {}\n".format(seed, recall_score(1-labels, 1-top_5_percent_preds)))
                f.write("Seed {} F1 {}\n".format(seed, f1_score(1-labels, 1-top_5_percent_preds)))
                f.write("Seed {} ROC AUC {}\n".format(seed, roc_auc_score(1-labels, anomaly_scores)))
                f.write("Seed {} Avg Prec {}\n".format(seed, average_precision_score(1-labels, anomaly_scores)))

        if full_covariance:
            with open('./results_gmms_seeds_full/precision_scores_{}_{}_mahal_default.txt'.format(args.data, best_k), 'a') as f:
                f.write("Seed {} Precision {}\n".format(seed, precision_score(1-labels, 1-preds_mahal_gmm_percentile)))
                # add recall and f1
                f.write("Seed {} Recall {}\n".format(seed, recall_score(1-labels, 1-preds_mahal_gmm_percentile)))
                f.write("Seed {} F1 {}\n".format(seed, f1_score(1-labels, 1-preds_mahal_gmm_percentile)))
                f.write("Seed {} ROC AUC {}\n".format(seed, roc_auc_score(1-labels, anomaly_scores_init)))
                f.write("Seed {} Avg Prec {}\n".format(seed, average_precision_score(1-labels, anomaly_scores_init)))
        else:
            with open('./results_gmms_seeds/precision_scores_{}_{}_mahal_default.txt'.format(args.data, args.save_suffix), 'a') as f:
                f.write("Seed {} Precision {}\n".format(seed, precision_score(1-labels, 1-preds_mahal_gmm_percentile)))
                # add recall and f1
                f.write("Seed {} Recall {}\n".format(seed, recall_score(1-labels, 1-preds_mahal_gmm_percentile)))
                f.write("Seed {} F1 {}\n".format(seed, f1_score(1-labels, 1-preds_mahal_gmm_percentile)))
                f.write("Seed {} ROC AUC {}\n".format(seed, roc_auc_score(1-labels, anomaly_scores_init)))
                f.write("Seed {} Avg Prec {}\n".format(seed, average_precision_score(1-labels, anomaly_scores_init)))

        
        print(">>>>__________<<<<")
    runtimes = np.array(runtimes)
    np.savetxt('./results_runtimes_gmms/runtimes_{}_{}.txt'.format(args.data, args.save_suffix), runtimes)

def evaluate_kmeans(features, k_range=range(2, 11), save_path="./results_kmeans", dataset_name="mydataset"):
    os.makedirs(save_path, exist_ok=True)
    
    # Optionally normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    inertias = []
    silhouettes = []

    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_scaled)

        inertia = kmeans.inertia_
        inertias.append(inertia)

        silhouette = silhouette_score(features_scaled, labels)
        silhouettes.append(silhouette)

        print(f"k={k} Inertia={inertia:.2f} Silhouette={silhouette:.4f}")

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(list(k_range), inertias, marker='o')
    ax[0].set_title("Elbow Method (Inertia)")
    ax[0].set_xlabel("Number of Clusters (k)")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(list(k_range), silhouettes, marker='o', color='green')
    ax[1].set_title("Silhouette Scores")
    ax[1].set_xlabel("Number of Clusters (k)")
    ax[1].set_ylabel("Silhouette Score")

    fig.suptitle(f"KMeans Evaluation for {dataset_name}", fontsize=14)
    plt.tight_layout()
    plot_file = os.path.join(save_path, f"{dataset_name}_kmeans_evaluation.png")
    plt.savefig(plot_file)
    print(f"Saved plot to: {plot_file}")
    plt.close()

def evaluate_k_range(features, k_range=range(2, 11), max_pca_dim=20, plot_path='./results_kmeans_reduced', random_state=42):
    """
    Evaluate multiple clustering metrics across a range of k values using KMeans.

    Parameters:
    - features (ndarray): Input data of shape (n_samples, n_features).
    - k_range (range or list): Values of k to evaluate.
    - max_pca_dim (int): Maximum PCA dimension to reduce to (for speed).
    - plot_path (str): Path to save plot of metrics. If None, doesn't save.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - best_k (int): Selected k based on max silhouette score.
    - metrics (dict): Contains scores for each k for each metric.
    """
    # Optional PCA for speed in high dimensions
    if features.shape[0] > 10000:
        X = shuffle(features, random_state=random_state)[:10000]
    else:
        X = features
    
    if X.shape[1] > max_pca_dim:
        pca = PCA(n_components=max_pca_dim, random_state=random_state)
        X = pca.fit_transform(X)

    ch_scores, db_scores, sil_scores = [], [], []

    for k in tqdm(k_range):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        sil_scores.append(silhouette_score(X, labels))

    metrics = {
        "k_values": list(k_range),
        "calinski_harabasz": ch_scores,
        "davies_bouldin": db_scores,
        "silhouette": sil_scores,
    }

    # Optional plotting
    if plot_path:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(k_range, ch_scores, marker='o')
        plt.title('Calinski-Harabasz (↑)')
        plt.xlabel("k")

        plt.subplot(1, 3, 2)
        plt.plot(k_range, db_scores, marker='o')
        plt.title('Davies-Bouldin (↓)')
        plt.xlabel("k")

        plt.subplot(1, 3, 3)
        plt.plot(k_range, sil_scores, marker='o')
        plt.title('Silhouette (↑)')
        plt.xlabel("k")

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # Best k based on sihouette score
    best_k = k_range[np.argmax(sil_scores)]
    #best_k = k_range[np.argmin(db_scores)]
    #best_k = k_range[np.argmax(ch_scores)]

    return best_k, metrics

if __name__ == "__main__":
    args = parse_arguments()
    #main(args)
    main_true(args)
    #main_boost(args)

   