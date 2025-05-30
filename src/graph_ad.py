import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse import csr_matrix, diags

from numpy.linalg import inv
from utils import *
import argparse
import time

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()
#from autoencoder import AE, train_model
#from torch.utils.data import TensorDataset, DataLoader
#import torch
np.random.seed(42)

import os


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='C:\\Users\\xdink\\Documents\\Research\\OnlineMetricLearning\\data')
    parser.add_argument('--data', type=str, default='census')
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--k', type=int, default=15)

    args = parser.parse_args()

    return args

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# def compute_directed_knn_graph(X, k, percentile=95):
#     """
#     Compute the weighted adjacency matrix W of a directed kNN graph.

#     Parameters:
#     - X : array of shape (n_samples, n_features)
#         Input data.
#     - k : int
#         Number of nearest neighbors.
#     - percentile : float
#         Percentile for estimating σ.

#     Returns:
#     - W : array of shape (n_samples, n_samples)
#         Weighted adjacency matrix.
#     """
#     n_samples = X.shape[0]

#     # Step 1: Fit kNN model
#     knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')  # k+1 because the point itself is included
#     knn.fit(X)
#     distances, indices = knn.kneighbors(X)

#     # Remove self-loop (distance to itself is 0)
#     distances = distances[:, 1:]
#     indices = indices[:, 1:]

#     # Step 2: Compute sigma
#     kth_distances = distances[:, -1]  # distance to k-th nearest neighbor for each point
#     sigma = 0.5 * np.percentile(kth_distances, percentile)

#     # Step 3: Build the weighted adjacency matrix
#     W = np.zeros((n_samples, n_samples))

#     for i in range(n_samples):
#         for j_idx, j in enumerate(indices[i]):
#             dist = distances[i, j_idx]
#             weight = np.exp(- (dist ** 2) / (2 * sigma ** 2))
#             W[i, j] = weight  # directed: only from i to j

#     return W

def build_sparse_weight_matrix(X, k=10, sigma=None):
    n = X.shape[0]
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    distances, indices = knn.kneighbors(X)

    # Optional: Use 95th percentile of kth distances to set sigma
    if sigma is None:
        kth_distances = distances[:, -1]
        sigma = 0.5 * np.percentile(kth_distances, 95)

    row, col, data = [], [], []

    for i in range(n):
        for j in range(1, k+1):  # skip j=0 because it's the point itself
            neighbor_idx = indices[i, j]
            dist = distances[i, j]
            weight = np.exp(- (dist ** 2) / (2 * sigma ** 2))
            row.append(i)
            col.append(neighbor_idx)
            data.append(weight)

    W = csr_matrix((data, (row, col)), shape=(n, n))
    return W, sigma

def compute_sparse_spreading_matrix(W):
    # Degree vector (row sums)
    d = np.array(W.sum(axis=1)).flatten()
    d_sqrt = np.sqrt(d)
    d_inv_sqrt = np.where(d_sqrt > 0, 1.0 / d_sqrt, 0.0)

    # Diagonal degree matrix as sparse
    D_inv_sqrt = diags(d_inv_sqrt)

    # Compute spreading matrix S = D^{-1/2} * W * D^{-1/2}
    S = D_inv_sqrt @ W @ D_inv_sqrt
    return S


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

class GAOD:
    def __init__(self, X, labels, k, alpha=0.95, budget=50, batch=1, seed=42):
        """
        X: np.ndarray
            The input data.
        W: np.ndarray
            The adjacency matrix of KNN graph.
        alpha: float
        budget: int
        batch: int
            The batch size.
        """
        self.original_X = X.copy()
        self.original_labels = labels.copy()
        self.X = X
        self.labels = labels
        self.k = k
        self.W, self.sigma = build_sparse_weight_matrix(X, k=k)
        self.alpha = alpha
        self.budget = budget
        self.remaining_budget = budget
        self.batch = batch
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.anomalies_found = 0
        self.seed = seed
        np.random.seed(seed)
    
    def compute_outlier_scores(self):
        """
        Compute the outlier score of iForest f_s
        """
        print("Computing Initial Outlier Scores...\n")
        iforest = IsolationForest(random_state=self.seed)
        iforest.fit(self.X)
        scores = iforest.score_samples(self.X)
        scores = normalize_data(scores)
        scores = 1-scores

        return scores, scores.min(), scores.max()
    
    
    def compute_spreading_matrix(self):
        """
        Compute the spreading matrix S
        """
        print("Computing Spreading Matrix...\n")
        #d = np.sum(self.W, axis=1)
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = np.sum(self.W[i])
        #d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        
        # Construct D^{-1/2}
        #D_inv_sqrt = np.diag(d_inv_sqrt)
        D_inv_sqrt = np.zeros((self.n, self.n))
        for i in range(self.n):
            if D[i, i] > 0:
                D_inv_sqrt[i, i] = 1.0 / np.sqrt(D[i, i])
            else:
                D_inv_sqrt[i, i] = 0.0  # Safe fallback for isolated nodes

        # Compute normalized matrix
        S = D_inv_sqrt @ self.W @ D_inv_sqrt
        return S
    
    def iterate(self):
        scores, scores_min, scores_max = self.compute_outlier_scores()
        # Compute the spreading matrix
        #S = self.compute_spreading_matrix()
        S = compute_sparse_spreading_matrix(self.W)
        alpha_vec = np.ones(self.n) * self.alpha
        f = scores
        f0 = scores.copy()
        queried = set()

        # while self.remaining_budget > 0:
        for _ in trange(self.remaining_budget, desc="Querying Budget"):
            masked_scores = f.copy()
            masked_scores[list(queried)] = -np.inf
            idx = np.argmax(masked_scores)
            # select top 1 unlabeled point
            #idx = np.argmax(f)
            point = self.X[idx]
            label = self.labels[idx]
            queried.add(idx)
            if label == 1:
                self.anomalies_found += 1
                f[idx] = scores_max
            else:
                f[idx] = scores_min
            alpha_vec[idx] = 1-self.alpha
            # update f until convergence
            for i in range(1000):
                f_new = alpha_vec * (S @ f) + (1 - alpha_vec) * f0
                diff = np.linalg.norm(f_new - f, ord=1)
                if diff < 1e-3:
                    break
                f = f_new

        self.remaining_budget = 0
        
        return f

def decision_function(f):
    # predict > 0.5 to be anomalous
    pred_labels = np.zeros(f.shape)
    pred_labels[f > 0.5] = 1
    pred_labels[f <= 0.5] = 0

    return pred_labels

def decision_function_95_percentile(f):
    # predict > 95th percentile of f to be anomalous
    threshold = np.percentile(f, 95)
    pred_labels = np.zeros(f.shape)
    pred_labels[f > threshold] = 1
    pred_labels[f <= threshold] = 0

    return pred_labels
    
def main(args):
    print(">>> Using Labels 1 for Anomaly, 0 for Nominal <<<\n")
    data, features, labels = resolve_data(args)
    features = remove_bad_features(features)
    binary_columns = [i for i in range(features.shape[1]) if np.unique(features[:, i]).size == 2]
    numeric_columns = [i for i in range(features.shape[1]) if i not in binary_columns]
    print("Number of Binary Columns {}".format(len(binary_columns)))

    if args.data != 'wine':
        #features, labels = smart_sampling(features=features, labels=labels, num_anoms=10, num_nominals=100)
        if args.data != 'bank':
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
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
    scaler = StandardScaler()

    numeric_features_scaled = scaler.fit_transform(features[:, numeric_columns])
    binary_features = features[:, binary_columns]
    features_stacked = np.hstack((numeric_features_scaled, binary_features))
    features = features_stacked
    ###########
    # shuffle the data
    features, labels = shuffle(features, labels, random_state=42)
    precs_reg = []
    recs_reg = []
    f1s_reg = []
    aucs_reg = []
    avg_precs_reg = []
    precs_95 = []
    recs_95 = []
    f1s_95 = []
    aucs_95 = []
    avg_precs_95 = []
    precs_iforest = []
    recs_iforest = []
    f1s_iforest = []
    aucs_iforest = []
    avg_precs_iforest = []
    anomalies_found = []
    runtimes = []
    for seed in np.arange(40, 50):
        start_time = time.time()
        gaod = GAOD(X=features, labels=labels, k=args.k, budget=args.budget, batch=args.k, seed=seed)
        scores = gaod.iterate()
        end_time = time.time()
        runtimes.append(end_time - start_time)
        precision_regular = precision_score(labels, decision_function(scores))
        recall_regular = recall_score(labels, decision_function(scores))
        f1_regular = f1_score(labels, decision_function(scores))
        auc_regular = roc_auc_score(labels, scores)
        avg_prec_reg = average_precision_score(labels, scores)
        precision_95 = precision_score(labels, decision_function_95_percentile(scores))
        recall_95 = recall_score(labels, decision_function_95_percentile(scores))
        f1_95 = f1_score(labels, decision_function_95_percentile(scores))
        auc_95 = roc_auc_score(labels, scores)
        avg_prec_95 = average_precision_score(labels, scores)
        precs_reg.append(precision_regular)
        recs_reg.append(recall_regular)
        f1s_reg.append(f1_regular)
        aucs_reg.append(auc_regular)
        avg_precs_reg.append(avg_prec_reg)
        precs_95.append(precision_95)
        recs_95.append(recall_95)
        f1s_95.append(f1_95)
        aucs_95.append(auc_95)
        avg_precs_95.append(avg_prec_95)
        anomalies_found.append(gaod.anomalies_found)
        print("Precision Regular: {}, Recall Regular: {}, F1 Regular: {} AUC Regular {} AvgPrec Regular {}".format(precision_regular, recall_regular, f1_regular, auc_regular, avg_prec_reg))
        print("Precision 95: {}, Recall 95: {}, F1 95: {} AUC 95 {} AvgPrec 95 {}".format(precision_95, recall_95, f1_95, auc_95, avg_prec_95))

        # default isolation forest scores
        iforest = IsolationForest(random_state=seed)
        iforest.fit(features)
        scores = iforest.score_samples(features)
        scores = normalize_data(scores)
        scores = 1-scores
        precision_iforest = precision_score(labels, decision_function(scores))
        recall_iforest = recall_score(labels, decision_function(scores))
        f1_iforest = f1_score(labels, decision_function(scores))
        auc_iforest = roc_auc_score(labels, scores)
        avg_prec_iforest = average_precision_score(labels, scores)
        precs_iforest.append(precision_iforest)
        recs_iforest.append(recall_iforest)
        f1s_iforest.append(f1_iforest)
        aucs_iforest.append(auc_iforest)
        avg_precs_iforest.append(avg_prec_iforest)
    print("Averages - Precision Regular: {}, Recall Regular: {}, F1 Regular: {} AUC Regular {} Avg Prec Regular {}".format(np.mean(precs_reg), np.mean(recs_reg), np.mean(f1s_reg), np.mean(aucs_reg), np.mean(avg_precs_reg)))
    print("Averages - Precision 95: {}, Recall 95: {}, F1 95: {} AUC 95 {} Avg Prec 95 {}".format(np.mean(precs_95), np.mean(recs_95), np.mean(f1s_95), np.mean(aucs_95), np.mean(avg_precs_95)))
    # overwrite and save prec/rec/f1 scores and avg anomalies found to csv file
    # in ./results
    save_path = "./results_gaod/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+args.data+".txt", 'w') as f:
        f.write(f"Precision {np.mean(precs_reg)} Recall {np.mean(recs_reg)} F1 {np.mean(f1s_reg)} AUC {np.mean(aucs_reg)} AvgPrec {np.mean(avg_precs_reg)} Avg Anomalies {np.mean(anomalies_found)}\n")
        f.write(f"Precision 95 {np.mean(precs_95)} Recall 95 {np.mean(recs_95)} F1 95 {np.mean(f1s_95)} AUC 95 {np.mean(aucs_95)} AvgPrec 95 {np.mean(avg_prec_95)} Avg Anomalies {np.mean(anomalies_found)}\n")
        f.write(f"Precision iForest {np.mean(precs_iforest)} Recall iForest {np.mean(recs_iforest)} F1 iForest {np.mean(f1s_iforest)} AUC iForest {np.mean(aucs_iforest)} AvgPrec iForest {np.mean(avg_prec_iforest)} Avg Anomalies {np.mean(anomalies_found)}\n")

    runtimes = np.array(runtimes)
    np.savetxt(save_path+args.data+"_runtimes.txt", runtimes, delimiter=",")

    


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
