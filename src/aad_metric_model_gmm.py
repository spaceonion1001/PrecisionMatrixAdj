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
from sklearn.utils import shuffle

from numpy.linalg import inv
from scipy.linalg import null_space
from scipy.special import expit

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

from scipy.stats import chi2

import torch
import torch.nn.functional as F

class AADMetricModel:
    def __init__(self, data, labels, v, J, init_dist_mat, mu, args, keep_top_k=False, top_k=3, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.data = data
        self.original_data = data.copy()
        self.labels = labels
        self.original_labels = labels.copy()
        self.v = v
        self.J = J

        self.curr_dist_mat = init_dist_mat # GMM specific initialization
        # self.keep_top_k = keep_top_k
        self.top_k = top_k
        self.args = args
        self.init_dist_mat = init_dist_mat
        self.queried_nominals = []
        self.queried_anomalies = []
        self.masked_data = np.ones(self.data.shape[0], dtype=bool)

        self.num_anomalies_queried = 0
        self.num_anomalies_queried_list = []

        self.queried_indices = []

        #self.a = chi2.ppf(0.95, self.data.shape[1]) #+ 2*np.sqrt(2*self.data.shape[1]) # significance level - 50 works well on imagenet for some reason
        self.a = 2
        self.mu = mu # GMM specific mean
    
    def iterate(self):
        w_s = []
        Z_s = []

        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.num_preds = []

        for j in range(self.J):
            # print(">>> ITER {} <<<".format(j))

            delta, curr_Z, is_anomaly, mu, u_point = self.get_user_feedback(args=self.args, use_mahal=False, use_threshold=False)
            all_queried_points = self.data[self.queried_indices, :]
            all_queried_labels = self.labels[self.queried_indices]
            if is_anomaly:
                z = np.linalg.svd(curr_Z)[0][:, 0]  # Take the first eigenvector of Z_t
                u = delta
                A_inv = np.linalg.inv(self.curr_dist_mat)
                M = (z[:, None] * A_inv) * z[None, :]
                Q = (u[:, None] * curr_Z) * u[None, :]
                # w = optimize_w_psd_projected_softplus(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, A_inv=A_inv, z=z, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=1000)
                w = optimize_w_fast(x_i=u_point, mu_X=mu, A_inv=A_inv, Z_t=curr_Z, M=M, Q=Q, v=self.v, p=self.data.shape[1], y=1, a=2.0, lr=1e-1, steps=20)
                A_t = self.curr_dist_mat + np.diag(w) @ curr_Z @ np.diag(w)
                w_s.append(w)
                Z_s.append(curr_Z)
                self.curr_dist_mat = A_t
            else:
                self.deweight_features(u_point=u_point, mean=mu)
                ###############################################################################
                ####
                # w = self.optimize_w_psd_projected_softplus(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=0, a=self.a, lr=1e-2, steps=1000, return_logs=False)
                # A_t = self.curr_dist_mat + np.diag(w) @ curr_Z @ np.diag(w)
                # w_s.append(w)
                # Z_s.append(curr_Z)
                # self.curr_dist_mat = A_t
                ####
            if not is_symmetric(self.curr_dist_mat):
                #print("Current distance matrix is not symmetric!")
                self.curr_dist_mat = (self.curr_dist_mat + self.curr_dist_mat.T) / 2.0
            assert is_pos_def(self.curr_dist_mat), "Current distance matrix is not positive definite! {}".format(np.linalg.eigvalsh(self.curr_dist_mat).min())
                

            self.num_anomalies_queried_list.append(self.num_anomalies_queried)
            # print()

        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        np.savetxt('figures/debugging_figs_true/num_anomalies_queried_{}_{}.csv'.format(self.args.data, self.args.save_suffix), np.array(self.num_anomalies_queried_list).astype(int))

        return self.curr_dist_mat
    
    def single_iterate(self, curr_Z, is_anomaly, u_point):
        # this only iterates a single point
        # updates the current distance matrix for one element of the mixture

        # delta, curr_Z, is_anomaly, mu, u_point = self.get_user_feedback(args=self.args, use_mahal=False, use_threshold=False)
        if is_anomaly:
            z = np.linalg.svd(curr_Z)[0][:, 0]  # Take the first eigenvector of Z_t
            u = u_point - self.mu
            A_inv = np.linalg.inv(self.curr_dist_mat)
            M = (z[:, None] * A_inv) * z[None, :]
            Q = (u[:, None] * curr_Z) * u[None, :]
            # w = optimize_w_psd_projected_softplus(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, A_inv=A_inv, z=z, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=1000)
            w = optimize_w_fast(x_i=u_point, mu_X=self.mu, A_inv=A_inv, Z_t=curr_Z, M=M, Q=Q, v=self.v, p=self.data.shape[1], y=1, a=2.0, lr=1e-1, steps=20)
            # w = optimize_w_psd_projected_softplus(x_i=u_point, mu_X=self.mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-1, steps=20)
            A_t = self.curr_dist_mat + np.diag(w) @ curr_Z @ np.diag(w)
            # w_s.append(w)
            # Z_s.append(curr_Z)
            self.curr_dist_mat = A_t
        else:
            self.deweight_features(u_point=u_point, mean=self.mu)
            ###############################################################################
            ####
            # w = self.optimize_w_psd_projected_softplus(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=0, a=self.a, lr=1e-2, steps=1000, return_logs=False)
            # A_t = self.curr_dist_mat + np.diag(w) @ curr_Z @ np.diag(w)
            # w_s.append(w)
            # Z_s.append(curr_Z)
            # self.curr_dist_mat = A_t
            ####
        if not is_symmetric(self.curr_dist_mat):
            #print("Current distance matrix is not symmetric!")
            self.curr_dist_mat = (self.curr_dist_mat + self.curr_dist_mat.T) / 2.0
        assert is_pos_def(self.curr_dist_mat), "Current distance matrix is not positive definite! {}".format(np.linalg.eigvalsh(self.curr_dist_mat).min())
            

        self.num_anomalies_queried_list.append(self.num_anomalies_queried)
        # print()
            
    def get_user_feedback(self, args, use_mahal=False, use_threshold=False):
        use_data = self.data[self.masked_data, :]
        use_labels = self.labels[self.masked_data]
        # print("Use Data Shape: ", use_data.shape)
        if not hasattr(self, "mean_vec"):
            self.mean_vec = np.mean(self.original_data, axis=0)
        mu = self.mean_vec
        anomaly_scores = calc_anomaly_scores(features=use_data, A_t=self.curr_dist_mat, mean_vec=mu)


        # greedy selection
        if not use_threshold:
            selected_idx = np.argmax(anomaly_scores)
            

        # select point just above than the 95th percentile - empirical
        else:
            above_threshold = anomaly_scores > np.percentile(anomaly_scores, 95)
            masked_scores = np.where(above_threshold, anomaly_scores, np.inf)
            selected_idx = np.argmin(masked_scores)

        # else:
            # inside_threshold = anomaly_scores < np.percentile(anomaly_scores, 95)
            # masked_scores = np.where(inside_threshold, anomaly_scores, -np.inf)
            # selected_idx = np.argmax(masked_scores)

        # select point just > than a
        # else:
        #     likely_anomalies = np.where((anomaly_scores**2) > self.a)[0]
        #     selected_idx = likely_anomalies[np.argmin(np.abs((anomaly_scores**2)[likely_anomalies] - self.a))]


        selected_score = anomaly_scores[selected_idx]
        selected_point = use_data[selected_idx, :]
        selected_label = use_labels[selected_idx]
        # print("Selected Point Score: {}".format(selected_score))
        # print("Selected Point Squared Score: {}".format(selected_score**2))
        #print("Score Mean: {}".format(np.mean(anomaly_scores)))
        #print("Score Std: {}".format(np.std(anomaly_scores)))
        # print("Squared Score Mean: {}".format(np.mean(anomaly_scores**2)))
        # print("Squared Score Std: {}".format(np.std(anomaly_scores**2)))
        # print("Squared Score Median: {}".format(np.median(anomaly_scores**2)))
       # print("Score Median: {}".format(np.median(anomaly_scores)))
        #print("Score 95th Percentile: {}".format(np.percentile(anomaly_scores, 95)))
        # print("Squared Score 95th Percentile: {}".format(np.percentile(anomaly_scores**2, 95)))
        #print("Score Max: {}".format(np.max(anomaly_scores)))
        #print("Score Min: {}".format(np.min(anomaly_scores)))
        # print("Squared Score Max: {}".format(np.max(anomaly_scores**2)))
        # print("Squared Score Min: {}".format(np.min(anomaly_scores**2)))
        # print("a value: {}".format(self.a))

        diff = selected_point - mu
        if use_mahal:
            #########
            eigvals, eigvecs = np.linalg.eigh(self.curr_dist_mat)
            # Construct L such that A = L^T L
            L = np.diag(np.sqrt(eigvals)) @ eigvecs.T
            v_whitened = L @ diff
            #v_whitened_unit = v_whitened / np.linalg.norm(v_whitened)
            mahal_grad = v_whitened
            ############

            # v_mahal = self.curr_dist_mat @ diff
            # mahal_grad = v_mahal
        else:
            mahal_grad = diff
        norm = np.linalg.norm(mahal_grad) + 1e-12
        z_vec = mahal_grad / norm
        z_vec = np.expand_dims(z_vec, 1)
        curr_Z = z_vec @ z_vec.T
        is_anomaly = (selected_label == 0)
        if selected_label == 0: # anomaly
            # print("Anomaly Found")
            self.num_anomalies_queried += 1
            self.queried_anomalies.append(selected_point)
            
        else:
            # print("Nominal Found")
            self.queried_nominals.append(selected_point)

        selected_indices = np.flatnonzero(self.masked_data)
        original_idx = selected_indices[selected_idx]
        # print("Removing Point {}".format(original_idx))
        self.queried_indices.append(original_idx)
        self.masked_data[original_idx] = False

        return diff, curr_Z, is_anomaly, mu, selected_point
    

    def optimize_w_psd_projected_softplus(self,
        x_i, mu_X, A_t_minus1, Z_t, v, p, y, a=2.0,
        lr=1e-2, steps=100, return_logs=False, leaky_constant=1
    ):
        u = x_i - mu_X
        d0 = u @ A_t_minus1 @ u
        Q = np.diag(u) @ Z_t @ np.diag(u)
        if y == 0:
            w = np.random.randn(p) * 0.01
        else:
            # w = np.random.randn(p) * 0.1
            w = np.ones(p)
        sqrt_2p = np.sqrt(2 * p)
        lambda_min = np.linalg.eigvalsh(A_t_minus1).min()

        # Compute D_z = diag(z) (where z is the eigenvector of Z_t)
        z = np.linalg.svd(Z_t)[0][:, 0]  # Take the first eigenvector of Z_t
        D_z = np.diag(z)
        # Compute M = D_z A_{t-1}^{-1} D_z
        A_inv = np.linalg.inv(A_t_minus1)
        M = D_z @ A_inv @ D_z

        if return_logs:
            logs = {"loss": [], "mahalanobis_sq": [], "deviation": []}
        first_loss = None
        first_dev = None
        for j in range(steps):
            quad = w @ Q @ w
            trace_quad = w @ Z_t @ w

            if y == 0:
                d = d0 - quad
                trace_term = -trace_quad
            else:
                d = d0 + quad
                trace_term = +trace_quad

            dev = (d - p) / sqrt_2p
            #dev = d
            trace_val = np.trace(A_t_minus1) + trace_term

            # Loss calculation (no penalty for the PSD constraint)
            if y == 0:
                loss = abs(dev) + v * trace_val
            else:
                #loss = np.log1p(np.exp(a - dev)) + v * trace_val
                # loss = np.log1p(np.exp(a - dev)) + v * trace_val + leaky_constant*(a-dev)
                loss = np.log1p(np.exp(a - dev)) + v * trace_val

            if return_logs:
                logs["loss"].append(loss)
                logs["mahalanobis_sq"].append(d)
                logs["deviation"].append(dev)

            # Gradient of dev
            if y == 0:
                grad_dev = -2 * Q @ w / sqrt_2p
                #grad_dev = -2 * Q @ w
                grad_trace = -2 * Z_t @ w
            else:
                grad_dev = +2 * Q @ w / sqrt_2p
                #grad_dev = +2 * Q @ w
                grad_trace = +2 * Z_t @ w

            # Final gradient
            if y == 0:
                grad_loss = np.sign(dev) * grad_dev + v * grad_trace
            else:
                # negative sigmoid for softplus
                #sigmoid = 1 / (1 + np.exp(-(a - dev)))
                # sigmoid = stable_sigmoid(a - dev)
                sigmoid = expit(a - dev)
                grad_loss = -sigmoid * grad_dev + v * grad_trace
                # grad_loss = -sigmoid * grad_dev + v * grad_trace - grad_dev*leaky_constant # with leaky version
                #grad_loss = -np.exp(a - dev)/(1 + np.exp(a - dev)) * grad_dev + v * grad_trace

            # Gradient step
            w -= lr * grad_loss

            # Projection to maintain the PSD constraint (for y = 0)
            if y == 0:
                

                # Check if the constraint is violated
                constraint_value = w @ M @ w
                if constraint_value > 1.0:
                    # Scale w to satisfy w^T M w = 1
                    scale = np.sqrt(1.0 / constraint_value)
                    w *= scale
            if j == 0:
                first_dev = dev
                first_loss = loss
        # print("First loss: ", first_loss)
        # print("Final loss: ", loss)
        # print("Loss Decreased: ", first_loss > loss)
        # print("First dev: ", first_dev)
        # print("Final dev: ", dev)
        return (w, logs) if return_logs else w
    


    
    def deweight_features(self, u_point, mean):
        diff = u_point - mean
        mahal_grad = self.curr_dist_mat @ diff
        contributions = np.abs(diff*mahal_grad)
        # get the top k contributions indices
        if self.args.full_k:
            # print("> Using Full K Deweighting... ")
            top_k_idx = np.arange(len(contributions))
        else:
            top_k_idx = np.argpartition(contributions, -self.top_k)[-self.top_k:]
        # if not self.args.use_top_k:
        #     top_k_idx = np.arange(len(contributions))
        eta = np.max(np.abs(contributions[top_k_idx]))  # Scale by max feature contribution
        alpha = np.clip(np.abs(contributions[top_k_idx]) / np.max(np.abs(contributions)), 0.1, 0.8)
        #alpha = 0.8
        #self.curr_dist_mat = low_rank_correction(self.curr_dist_mat, top_k_idx, eta=eta)
        if self.args.eigval_deweight:
            self.curr_dist_mat = eigenvalue_softening(self.curr_dist_mat, top_k_idx, alpha=alpha)
        #self.curr_dist_mat = hybrid_precision_update(self.curr_dist_mat, top_k_idx, eta=eta, alpha=alpha)
        else:
            v_mahal = self.curr_dist_mat @ diff
            v_mahal_unit = v_mahal / (np.linalg.norm(v_mahal) + 1e-12)
            eigvals, eigvecs = np.linalg.eigh(self.curr_dist_mat)
            # Construct L such that A = L^T L
            L = np.diag(np.sqrt(eigvals)) @ eigvecs.T
            v_whitened = L @ diff
            v_whitened_unit = v_whitened / np.linalg.norm(v_whitened)
            self.curr_dist_mat = w_precision_update(self.curr_dist_mat, top_k_idx, direction_vector=diff)
            #self.curr_dist_mat = w_precision_update(self.curr_dist_mat, top_k_idx, direction_vector=v_whitened_unit)


def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis_fast_uv(v, mean_vec, A_t)
        dists.append(curr_dist)

    return np.array(dists)

def stable_sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

# @njit
# def optimize_w_psd_projected_softplus(
#         x_i, mu_X, A_t_minus1, Z_t, v, p, y, 
#         z, A_inv, a=2.0,
#         lr=1e-2, steps=100
#     ):
#     u = x_i - mu_X
#     d0 = u @ A_t_minus1 @ u
#     # Q = np.diag(u) @ Z_t @ np.diag(u)
#     Q = (u[:, None] * Z_t) * u[None, :]
#     if y == 0:
#         w = np.random.randn(p) * 0.01
#     else:
#         # w = np.random.randn(p) * 0.1
#         w = np.ones(p)
#     sqrt_2p = np.sqrt(2 * p)
#     # lambda_min = np.linalg.eigvalsh(A_t_minus1).min()

#     # Compute D_z = diag(z) (where z is the eigenvector of Z_t)
#     #D_z = np.diag(z)
#     # Compute M = D_z A_{t-1}^{-1} D_z
#     #M = D_z @ A_inv @ D_z
#     M = (z[:, None] * A_inv) * z[None, :]

#     first_loss = None
#     first_dev = None
#     for j in range(steps):
#         quad = w @ Q @ w
#         trace_quad = w @ Z_t @ w

#         if y == 0:
#             d = d0 - quad
#             trace_term = -trace_quad
#         else:
#             d = d0 + quad
#             trace_term = +trace_quad

#         dev = (d - p) / sqrt_2p
#         #dev = d
#         trace_val = np.trace(A_t_minus1) + trace_term

#         # Loss calculation (no penalty for the PSD constraint)
#         if y == 0:
#             loss = abs(dev) + v * trace_val
#         else:
#             #loss = np.log1p(np.exp(a - dev)) + v * trace_val
#             # loss = np.log1p(np.exp(a - dev)) + v * trace_val + leaky_constant*(a-dev)
#             loss = np.log1p(np.exp(a - dev)) + v * trace_val

#         # Gradient of dev
#         if y == 0:
#             grad_dev = -2 * Q @ w / sqrt_2p
#             #grad_dev = -2 * Q @ w
#             grad_trace = -2 * Z_t @ w
#         else:
#             grad_dev = +2 * Q @ w / sqrt_2p
#             #grad_dev = +2 * Q @ w
#             grad_trace = +2 * Z_t @ w

#         # Final gradient
#         if y == 0:
#             grad_loss = np.sign(dev) * grad_dev + v * grad_trace
#         else:
#             # negative sigmoid for softplus
#             #sigmoid = 1 / (1 + np.exp(-(a - dev)))
#             # sigmoid = stable_sigmoid(a - dev)
#             sigmoid = expit(a - dev)
#             grad_loss = -sigmoid * grad_dev + v * grad_trace
#             # grad_loss = -sigmoid * grad_dev + v * grad_trace - grad_dev*leaky_constant # with leaky version
#             #grad_loss = -np.exp(a - dev)/(1 + np.exp(a - dev)) * grad_dev + v * grad_trace

#         # Gradient step
#         w -= lr * grad_loss

#         # Projection to maintain the PSD constraint (for y = 0)
#         if y == 0:
            

#             # Check if the constraint is violated
#             constraint_value = w @ M @ w
#             if constraint_value > 1.0:
#                 # Scale w to satisfy w^T M w = 1
#                 scale = np.sqrt(1.0 / constraint_value)
#                 w *= scale
#         if j == 0:
#             first_dev = dev
#             first_loss = loss
#     # print("First loss: ", first_loss)
#     # print("Final loss: ", loss)
#     # print("Loss Decreased: ", first_loss > loss)
#     # print("First dev: ", first_dev)
#     # print("Final dev: ", dev)
#     return w

@njit
def optimize_w_fast(x_i, mu_X, A_inv, Z_t, M, Q, v, p, y, a=2.0, lr=1e-2, steps=100):
    u = x_i - mu_X
    d0 = u @ A_inv @ u
    w = np.ones(p) if y == 1 else np.random.randn(p) * 0.01
    sqrt_2p = np.sqrt(2 * p)

    for j in range(steps):
        Qw = Q @ w
        Zw = Z_t @ w
        quad = w @ Qw
        trace_quad = w @ Zw

        if y == 0:
            d = d0 - quad
            trace_val = np.trace(A_inv) - trace_quad
            dev = (d - p) / sqrt_2p
            # loss = np.abs(dev) + v * trace_val
            grad_dev = -2 * Qw / sqrt_2p
            grad_trace = -2 * Zw
            grad_loss = np.sign(dev) * grad_dev + v * grad_trace
        else:
            d = d0 + quad
            trace_val = np.trace(A_inv) + trace_quad
            dev = (d - p) / sqrt_2p
            # loss = np.log1p(np.exp(a - dev)) + v * trace_val
            # sigmoid = 1 / (1 + np.exp(-(a - dev)))  # or use expit
            sigmoid = expit(a - dev)
            grad_dev = +2 * Qw / sqrt_2p
            grad_trace = +2 * Zw
            grad_loss = -sigmoid * grad_dev + v * grad_trace

        w -= lr * grad_loss

        if y == 0:
            temp = M @ w
            constraint_value = w @ temp
            if constraint_value > 1.0:
                w *= np.sqrt(1.0 / constraint_value)

    return w


    