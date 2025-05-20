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
    def __init__(self, data, labels, v, J, init_dist_mat, args, keep_top_k=False, top_k=3, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.data = data
        self.original_data = data.copy()
        self.labels = labels
        self.original_labels = labels.copy()
        self.v = v
        self.J = J

        self.curr_dist_mat = init_dist_mat
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
    
    def iterate(self):
        w_s = []
        Z_s = []

        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.num_preds = []

        for j in range(self.J):
            print(">>> ITER {} <<<".format(j))

            delta, curr_Z, is_anomaly, mu, u_point = self.get_user_feedback(args=self.args, use_mahal=False, use_threshold=False)
            all_queried_points = self.data[self.queried_indices, :]
            all_queried_labels = self.labels[self.queried_indices]
            if is_anomaly:
                #w, A_t = self.optimize_w(delta=delta, Z_t=curr_Z, A_prev=self.curr_dist_mat, y=1, a=self.a, v=self.v, lr=1e-2, max_iters=100, tol=1e-5, verbose=True)
                #w = self.optimize_w_numpy_verbose(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=100, return_logs=False)
                #w = self.optimize_w_label_dependent_update(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=100, return_logs=False)
                #w = self.optimize_w_psd_projected(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=1000, return_logs=False)

                w = self.optimize_w_psd_projected_softplus(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=1, a=self.a, lr=1e-2, steps=1000, return_logs=False)
                # w_norm = 2 * (w - w.min()) / (w.max() - w.min()) - 1

                #w = self.optimize_logcosh(u=delta, Z_t=curr_Z, a=self.a, beta=0.5, lr=0.01, steps=1000, verbose=False)
                # w = self.optimize_piecewise_power_decay(u=delta, A_t_minus_1=self.curr_dist_mat, Z_t=curr_Z, a=self.a,
                #                     p=1.0, v=1e-3, eps=1e-6, alpha=1.0,
                #                     lr=0.01, steps=1000, verbose=False)
                #w = self.optimize_w_torch_softplus(u=delta, A_t_minus_1=self.curr_dist_mat, Z_t=curr_Z, a=self.a, y=1, v=1e-3, lr=1e-1, steps=500)
                
                # w = optimize_w_psd_projected_softplus_batch(x_i=u_point, x_list=all_queried_points, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, 
                #                                                  p=self.data.shape[1], y_i=1, y_list=all_queried_labels, 
                #                                                  a=self.a, lr=1e-2, steps=500)
                #############
                A_t = self.curr_dist_mat + np.diag(w) @ curr_Z @ np.diag(w)
                # A_t = self.curr_dist_mat + np.diag(w_norm) @ curr_Z @ np.diag(w_norm)
                w_s.append(w)
                Z_s.append(curr_Z)
                self.curr_dist_mat = A_t
            else:
                self.deweight_features(u_point=u_point, mean=mu)
                ###############################################################################
                #w = self.optimize_w_numpy_verbose(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=0, a=self.a, lr=1e-2, steps=100, return_logs=False)
                #w = self.optimize_w_psd_projected(x_i=u_point, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, p=self.data.shape[1], y=0, a=self.a, lr=1e-2, steps=1000, return_logs=False)
                ####
                ####
                #w = self.optimize_w_torch_softplus(u=delta, A_t_minus_1=self.curr_dist_mat, Z_t=curr_Z, a=self.a, y=0, v=1e-3, lr=1e-1, steps=500)
                # w = optimize_w_psd_projected_softplus_batch(x_i=u_point, x_list=all_queried_points, mu_X=mu, A_t_minus1=self.curr_dist_mat, Z_t=curr_Z, v=self.v, 
                #                                                  p=self.data.shape[1], y_i=0, y_list=all_queried_labels, 
                #                                                  a=self.a, lr=1e-2, steps=500)
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
            print()

        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        np.savetxt('figures/debugging_figs_true/num_anomalies_queried_{}_{}.csv'.format(self.args.data, self.args.save_suffix), np.array(self.num_anomalies_queried_list).astype(int))

        return self.curr_dist_mat
            


    def get_user_feedback(self, args, use_mahal=False, use_threshold=False):
        use_data = self.data[self.masked_data, :]
        use_labels = self.labels[self.masked_data]
        print("Use Data Shape: ", use_data.shape)
        mu = np.mean(self.original_data, axis=0)
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
        print("Selected Point Score: {}".format(selected_score))
        print("Selected Point Squared Score: {}".format(selected_score**2))
        #print("Score Mean: {}".format(np.mean(anomaly_scores)))
        #print("Score Std: {}".format(np.std(anomaly_scores)))
        print("Squared Score Mean: {}".format(np.mean(anomaly_scores**2)))
        print("Squared Score Std: {}".format(np.std(anomaly_scores**2)))
        print("Squared Score Median: {}".format(np.median(anomaly_scores**2)))
       # print("Score Median: {}".format(np.median(anomaly_scores)))
        #print("Score 95th Percentile: {}".format(np.percentile(anomaly_scores, 95)))
        print("Squared Score 95th Percentile: {}".format(np.percentile(anomaly_scores**2, 95)))
        #print("Score Max: {}".format(np.max(anomaly_scores)))
        #print("Score Min: {}".format(np.min(anomaly_scores)))
        print("Squared Score Max: {}".format(np.max(anomaly_scores**2)))
        print("Squared Score Min: {}".format(np.min(anomaly_scores**2)))
        print("a value: {}".format(self.a))

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
        u_vec = mahal_grad/np.linalg.norm(mahal_grad)
        coef_abs = np.abs(u_vec)
        z_vec = u_vec
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        #print(z_vec)
        curr_Z = z_vec@z_vec.T
        is_anomaly = (selected_label == 0)
        if selected_label == 0: # anomaly
            print("Anomaly Found")
            self.num_anomalies_queried += 1
            self.queried_anomalies.append(selected_point)
            
        else:
            print("Nominal Found")
            self.queried_nominals.append(selected_point)

        selected_indices = np.flatnonzero(self.masked_data)
        original_idx = selected_indices[selected_idx]
        print("Removing Point {}".format(original_idx))
        self.queried_indices.append(original_idx)
        self.masked_data[original_idx] = False

        return diff, curr_Z, is_anomaly, mu, selected_point
    
    def mahalanobis_loss_and_grad(self, w, delta, Z_t, A_prev, y, a, v):
        """
        Compute loss and gradient with respect to w.
        """
        d = len(w)
        D_w = np.diag(w)
        B_w = D_w @ Z_t @ D_w
        A_t = A_prev + B_w

        delta = delta.reshape(-1, 1)  # (d, 1)
        d_squared = float(delta.T @ A_t @ delta)
        d_val = np.sqrt(d_squared + 1e-10)  # stability
        print("d_val: ", d_val)

        # Loss
        dist_term = (1 - y) * d_val + y * max(0, a - d_val)
        reg_term = v * (w @ Z_t @ w)
        loss = dist_term + reg_term

        # Gradient
        u = w * delta.flatten()  # diag(w) @ delta
        Z_u = Z_t @ u            # (d,)
        d_s = 2 * Z_u * delta.flatten()  # ∇_w s
        coeff = ((1 - y) - y * (d_val < a)) / (2 * d_val)
        grad = coeff * d_s + 2 * v * (Z_t @ w)
        print(loss, grad, A_t)
        exit()

        return loss, grad, A_t

    def optimize_w(self, delta, Z_t, A_prev, y, a, v, lr=1e-2, max_iters=100, tol=1e-6, verbose=False):
        """
        Optimize w using gradient descent.
        """
        d = delta.shape[0]
        #w = np.zeros(d)
        w = np.random.normal(scale=0.01, size=d)
        for i in range(max_iters):
            loss, grad, A_t = self.mahalanobis_loss_and_grad(w, delta, Z_t, A_prev, y, a, v)
            w_new = w - lr * grad
            if np.linalg.norm(w_new - w) < tol:
                if verbose:
                    print(f"Converged at iter {i}")
                break
            w = w_new
            if verbose and i % 10 == 0:
                print(f"Iter {i}: Loss = {loss:.4f}")
        return w, A_t

    # def optimize_w_label_dependent_update(
    #     x_i, mu_X, A_t_minus1, Z_t, v, p, y, a=2.0,
    #     lr=1e-2, steps=100, return_logs=True
    # ):
    #     """
    #     Optimize w where A_t = A_{t-1} ± diag(w) Z_t diag(w)
    #     Subtractive update for y=0 (normal), additive for y=1 (anomaly)
    #     """
    #     u = x_i - mu_X
    #     d0 = u @ A_t_minus1 @ u
    #     Q = np.diag(u) @ Z_t @ np.diag(u)
    #     w = np.random.randn(p) * 0.01
    #     sqrt_2p = np.sqrt(2 * p)

    #     if return_logs:
    #         logs = {"loss": [], "mahalanobis_sq": [], "deviation": []}

    #     for _ in range(steps):
    #         quad = w @ Q @ w
    #         trace_quad = w @ Z_t @ w

    #         if y == 0:
    #             d = d0 - quad
    #             trace_term = -trace_quad
    #         else:
    #             d = d0 + quad
    #             trace_term = +trace_quad

    #         dev = (d - p) / sqrt_2p
    #         trace_val = np.trace(A_t_minus1) + trace_term

    #         # Loss
    #         if y == 0:
    #             loss = abs(dev) + v * trace_val
    #         else:
    #             loss = max(0.0, a - dev) + v * trace_val

    #         if return_logs:
    #             logs["loss"].append(loss)
    #             logs["mahalanobis_sq"].append(d)
    #             logs["deviation"].append(dev)

    #         # Gradient of dev wrt w
    #         if y == 0:
    #             grad_dev = -2 * Q @ w / sqrt_2p
    #             grad_trace = -2 * Z_t @ w
    #         else:
    #             grad_dev = +2 * Q @ w / sqrt_2p
    #             grad_trace = +2 * Z_t @ w

    #         # Full gradient of loss
    #         if y == 0:
    #             grad_loss = np.sign(dev) * grad_dev + v * grad_trace
    #         else:
    #             if dev < a:
    #                 grad_margin = -grad_dev
    #             else:
    #                 grad_margin = 0.0
    #             grad_loss = grad_margin + v * grad_trace

    #         # Gradient step
    #         w -= lr * grad_loss

    #     return (w, logs) if return_logs else w

    # def optimize_w_psd_projected(self,
    #     x_i, mu_X, A_t_minus1, Z_t, v, p, y, a=2.0,
    #     lr=1e-2, steps=100, return_logs=True
    # ):
    #     u = x_i - mu_X
    #     d0 = u @ A_t_minus1 @ u
    #     Q = np.diag(u) @ Z_t @ np.diag(u)
    #     w = np.random.randn(p) * 0.01
    #     sqrt_2p = np.sqrt(2 * p)
    #     lambda_min = np.linalg.eigvalsh(A_t_minus1).min()

    #     # Compute D_z = diag(z) (where z is the eigenvector of Z_t)
    #     z = np.linalg.svd(Z_t)[0][:, 0]  # Take the first eigenvector of Z_t
    #     D_z = np.diag(z)

    #     if return_logs:
    #         logs = {"loss": [], "mahalanobis_sq": [], "deviation": []}

    #     for _ in range(steps):
    #         quad = w @ Q @ w
    #         trace_quad = w @ Z_t @ w

    #         if y == 0:
    #             d = d0 - quad
    #             trace_term = -trace_quad
    #         else:
    #             d = d0 + quad
    #             trace_term = +trace_quad

    #         dev = (d - p) / sqrt_2p
    #         trace_val = np.trace(A_t_minus1) + trace_term

    #         # Loss calculation (no penalty for the PSD constraint)
    #         if y == 0:
    #             loss = abs(dev) + v * trace_val
    #         else:
    #             loss = max(0.0, a - dev) + v * trace_val

    #         if return_logs:
    #             logs["loss"].append(loss)
    #             logs["mahalanobis_sq"].append(d)
    #             logs["deviation"].append(dev)

    #         # Gradient of dev
    #         if y == 0:
    #             grad_dev = -2 * Q @ w / sqrt_2p
    #             grad_trace = -2 * Z_t @ w
    #         else:
    #             grad_dev = +2 * Q @ w / sqrt_2p
    #             grad_trace = +2 * Z_t @ w

    #         # Final gradient
    #         if y == 0:
    #             grad_loss = np.sign(dev) * grad_dev + v * grad_trace
    #         else:
    #             if dev < a:
    #                 grad_margin = -grad_dev
    #             else:
    #                 grad_margin = 0.0
    #             grad_loss = grad_margin + v * grad_trace

    #         # Gradient step
    #         w -= lr * grad_loss

    #         # Projection to maintain the PSD constraint (for y = 0)
    #         if y == 0:
    #             # Compute M = D_z A_{t-1}^{-1} D_z
    #             A_inv = np.linalg.inv(A_t_minus1)
    #             M = D_z @ A_inv @ D_z

    #             # Check if the constraint is violated
    #             constraint_value = w @ M @ w
    #             if constraint_value > 1.0:
    #                 # Scale w to satisfy w^T M w = 1
    #                 scale = np.sqrt(1.0 / constraint_value)
    #                 w *= scale
    #     print("Final loss: ", loss)
    #     print("Final dev: ", dev)
    #     return (w, logs) if return_logs else w
    
    def optimize_w_psd_projected_softplus(self,
        x_i, mu_X, A_t_minus1, Z_t, v, p, y, a=2.0,
        lr=1e-2, steps=100, return_logs=True, leaky_constant=1
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
                sigmoid = 1 / (1 + np.exp(-(a - dev)))
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
        print("First loss: ", first_loss)
        print("Final loss: ", loss)
        print("Loss Decreased: ", first_loss > loss)
        print("First dev: ", first_dev)
        print("Final dev: ", dev)
        return (w, logs) if return_logs else w
    

    def optimize_w_psd_projected_softplus_batch(self, x_i,
        x_list, mu_X, A_t_minus1, Z_t, v, p, y_i, y_list, a=2.0,
        lr=1e-2, steps=100, return_logs=True
    ):
        if y_i == 0:
            w = np.random.randn(p) * 0.01
        else:
            w = np.ones(p)
        # Compute D_z = diag(z) (where z is the eigenvector of Z_t)
        z = np.linalg.svd(Z_t)[0][:, 0]  # Take the first eigenvector of Z_t
        D_z = np.diag(z)

        if return_logs:
            logs = {"loss": [], "mahalanobis_sq": [], "deviation": []}
        first_loss = None
        first_dev = None
        for j in range(steps):
            curr_grad_loss = 0
            curr_loss = 0
            for i in range(len(x_list)):
                x_curr = x_list[i]
                y_curr = y_list[i]
                u = x_curr - mu_X
                d0 = u @ A_t_minus1 @ u
                Q = np.diag(u) @ Z_t @ np.diag(u)
                quad = w @ Q @ w
                trace_quad = w @ Z_t @ w

                if y_i == 0:
                    d = d0 - quad
                    trace_term = -trace_quad
                else:
                    d = d0 + quad
                    trace_term = +trace_quad

                dev = d
                trace_val = np.trace(A_t_minus1) + trace_term

                # Loss calculation (no penalty for the PSD constraint)
                if y_curr == 0:
                    loss = abs(dev) + v * trace_val
                else:
                    #loss = max(0.0, a - dev) + v * trace_val
                    loss = np.log1p(np.exp(a - dev)) + v * trace_val

                if return_logs:
                    logs["loss"].append(loss)
                    logs["mahalanobis_sq"].append(d)
                    logs["deviation"].append(dev)

                # Gradient of dev
                if y_i == 0:
                    grad_dev = -2 * Q @ w
                    grad_trace = -2 * Z_t @ w
                else:
                    grad_dev = +2 * Q @ w
                    grad_trace = +2 * Z_t @ w

                # Final gradient
                if y_curr == 0:
                    grad_loss = np.sign(dev) * grad_dev + v * grad_trace
                else:
                    # negative sigmoid for softplus
                    grad_loss = -np.exp(a - dev)/(1 + np.exp(a - dev)) * grad_dev + v * grad_trace
                curr_grad_loss += grad_loss
                curr_loss += loss

            # Gradient step
            w -= lr * curr_grad_loss

            # Projection to maintain the PSD constraint (for y = 0)
            # Compute M = D_z A_{t-1}^{-1} D_z
            A_inv = np.linalg.inv(A_t_minus1)
            M = D_z @ A_inv @ D_z

            # Check if the constraint is violated
            constraint_value = w @ M @ w
            if constraint_value > 1.0:
                # Scale w to satisfy w^T M w = 1
                scale = np.sqrt(1.0 / constraint_value)
                w *= scale
            if j == 0:
                first_loss = curr_loss
        print("First loss: ", first_loss)
        print("Final loss: ", curr_loss)
        print("Loss Decreased: ", first_loss > curr_loss)
        return (w, logs) if return_logs else w
    
    def optimize_w_psd_projected_softplus_batch_vectorized(self,
        x_i, x_list, mu_X, A_t_minus1, Z_t, v, p, y_i, y_list, a=2.0, lr=1e-2, steps=100
    ):
        if y_i == 0:
            w = np.random.randn(p) * 0.01
        else:
            w = np.ones(p)

        x_array = np.stack(x_list)  # shape: (N, p)
        u_array = x_array - mu_X  # shape: (N, p)
        y_array = np.array(y_list)  # shape: (N,)

        D_z = np.diag(np.linalg.svd(Z_t)[0][:, 0])
        A_inv = np.linalg.inv(A_t_minus1)
        M = D_z @ A_inv @ D_z

        first_loss = None

        for j in range(steps):
            d0_all = np.einsum("np, pq, nq -> n", u_array, A_t_minus1, u_array)  # shape: (N,)

            u_diag = np.transpose(u_array[:, :, None] * np.eye(p), (0, 2, 1))  # shape: (N, p, p)
            Q_all = np.matmul(np.matmul(u_diag, Z_t), u_diag)  # shape: (N, p, p)

            Qw_all = np.matmul(Q_all, w)  # shape: (N, p)
            quad_all = np.einsum("np, np -> n", w[None, :] * np.ones_like(Qw_all), Qw_all)  # shape: (N,)
            trace_quad_all = w @ Z_t @ w

            if y_i == 0:
                d_all = d0_all - quad_all
                trace_term_all = -trace_quad_all
            else:
                d_all = d0_all + quad_all
                trace_term_all = +trace_quad_all

            trace_val = np.trace(A_t_minus1) + trace_term_all

            # Compute losses
            loss_array = np.where(
                y_array == 0,
                np.abs(d_all) + v * trace_val,
                np.log1p(np.exp(a - d_all)) + v * trace_val,
            )
            curr_loss = np.sum(loss_array)

            # Compute gradients
            grad_dev_all = np.where(
                y_i == 0,
                -2 * Qw_all,
                2 * Qw_all,
            )
            grad_trace = (-2 if y_i == 0 else 2) * Z_t @ w

            grad_loss_all = np.where(
                y_array[:, None] == 0,
                np.sign(d_all)[:, None] * grad_dev_all,
                (-np.exp(a - d_all) / (1 + np.exp(a - d_all)))[:, None] * grad_dev_all,
            )
            grad_loss_all += v * grad_trace

            curr_grad_loss = grad_loss_all.sum(axis=0)
            w -= lr * curr_grad_loss

            constraint_value = w @ M @ w
            if constraint_value > 1.0:
                scale = np.sqrt(1.0 / constraint_value)
                w *= scale

            if j == 0:
                first_loss = curr_loss

        print("First loss:", first_loss)
        print("Final loss:", curr_loss)
        print("Loss Decreased:", first_loss > curr_loss)
        return w
    
    def logcosh_loss_and_grad(self, w, u, Z_t, a, beta=1.0, v=1e-3):
        """
        Log-cosh loss and its gradient w.r.t. w.

        Parameters:
            w:     (d,) parameter vector
            u:     (d,) input vector (e.g., x_i - mu_X)
            Z_t:   (d, d) matrix (symmetric)
            a:     scalar threshold
            beta:  scale parameter

        Returns:
            loss: scalar loss value
            grad_w: (d,) gradient w.r.t. w
        """
        D = np.diag(u)
        DZDt = D @ Z_t @ D
        f = w.T @ DZDt @ w

        trace_term = w @ Z_t @ w
        grad_trace = 2 * Z_t @ w
        
        delta = beta * (a - f)
        #loss = np.log(np.cosh(delta)) + v * trace_term
        loss = self.safe_logcosh(delta) + v * trace_term
        
        grad_scalar = -beta * np.tanh(delta)
        grad_w = grad_scalar * 2 * DZDt @ w + grad_trace * v
        return loss, grad_w

    # === Optimization ===

    def optimize_logcosh(self, u, Z_t, a, beta=1.0, lr=0.01, steps=200, verbose=False):
        #w = w_init.copy()
        w = np.ones(Z_t.shape[0])
        starting_loss = self.logcosh_loss_and_grad(w, u, Z_t, a, beta)[0]
        for step in range(steps):
            loss, grad = self.logcosh_loss_and_grad(w, u, Z_t, a, beta)
            w -= lr * grad
            if verbose and step % 20 == 0:
                print(f"Step {step:3d} | Loss: {loss:.6f} | ||grad||: {np.linalg.norm(grad):.4f}")
        final_loss = self.logcosh_loss_and_grad(w, u, Z_t, a, beta)[0]
        print("Starting loss: ", starting_loss)
        print("Final loss: ", final_loss)
        print("Loss Decreased: ", starting_loss > final_loss)
        return w

    def safe_logcosh(self, x):
        abs_x = np.abs(x)
        return abs_x + np.log1p(np.exp(-2 * abs_x)) - np.log(2)
    
    # def piecewise_power_decay_loss_and_grad(self, w, u, A_t_minus_1, Z_t, a, p=1.0, alpha=1.0, v=0.0, eps=1e-6):
    #     D = np.diag(u)
    #     DZDt = D @ Z_t @ D
    #     f = u.T @ A_t_minus_1 @ u + w.T @ DZDt @ w

    #     if f < a:
    #         loss = a - f
    #         grad_w = -2 * DZDt @ w
    #     else:
    #         decay = alpha / (f - a + eps)**p
    #         print("f: ", f)
    #         print("a: ", a)
    #         print("eps: ", eps)
    #         loss = decay
    #         grad_scalar = -alpha * p / (f - a + eps)**(p + 1)
    #         grad_w = grad_scalar * 2 * DZDt @ w

    #     trace_reg = np.trace(A_t_minus_1) + np.sum(np.diag(DZDt) * w**2)
    #     loss += v * trace_reg
    #     grad_w += v * 2 * np.diag(DZDt) * w

    #     return loss, grad_w
    
    # def optimize_piecewise_power_decay(self, u, A_t_minus_1, Z_t, a,
    #                                 p=1.0, v=0.0, eps=1e-6, alpha=10.0,
    #                                 lr=0.01, steps=200, verbose=False):
    #     """
    #     Optimize w using the piecewise power decay loss.
    #     """
    #     #w = w_init.copy()
    #     w = np.ones(Z_t.shape[0])
    #     starting_loss = self.piecewise_power_decay_loss_and_grad(w=w, u=u, A_t_minus_1=A_t_minus_1, Z_t=Z_t, a=a, p=p, alpha=alpha, v=v, eps=eps)[0]
    #     for step in range(steps):
    #         loss, grad = self.piecewise_power_decay_loss_and_grad(
    #             w=w, u=u, A_t_minus_1=A_t_minus_1, Z_t=Z_t, a=a, p=a, alpha=alpha, v=v, eps=eps
    #         )
    #         w -= lr * grad
    #         if verbose and step % 20 == 0:
    #             print(f"Step {step:3d} | Loss: {loss:.6f} | ||grad||: {np.linalg.norm(grad):.4f}")
    #     final_loss = self.piecewise_power_decay_loss_and_grad(w=w, u=u, A_t_minus_1=A_t_minus_1, Z_t=Z_t, a=a, p=p, alpha=alpha, v=v, eps=eps)[0]
    #     print("Starting loss: ", starting_loss)
    #     print("Final loss: ", final_loss)
    #     print("Loss Decreased: ", starting_loss > final_loss)
    #     return w

    

    def softplus_loss(self, w, u, A_t_minus_1, Z_t, a, v):
        """
        w: (d,)  - requires_grad=True
        u: (d,)  - fixed
        A_t_minus_1: (d,d) - fixed
        Z_t: (d,d) - fixed
        a: scalar - target threshold
        v: scalar - trace regularization weight
        """
        D = torch.diag(u)                             # diag(u)
        DZDt = D @ Z_t @ D                            # diag(u) Z_t diag(u)
        f = u @ (A_t_minus_1 @ u) + w @ (DZDt @ w)    # f(x)

        trace_At = torch.trace(A_t_minus_1 + torch.diag(w) @ Z_t @ torch.diag(w))
        loss = F.softplus(a - f) + v * trace_At

        return loss
    

    def piecewise_loss(self, w, u, A_t_minus_1, Z_t, a, v, y):
        """
        Parameters:
            w: (d,)               - requires_grad=True
            u: (d,)               - vector (x_i - mu)
            A_t_minus_1: (d, d)   - PSD matrix
            Z_t: (d, d)           - symmetric matrix
            a: float              - target threshold
            v: float              - trace penalty weight
            y: int (0 or 1)       - class label
        Returns:
            loss: scalar
        """
        D = torch.diag(u)
        DZDt = D @ Z_t @ D
        f = u @ (A_t_minus_1 @ u) + w @ (DZDt @ w)

        if y == 1:
            A_t = A_t_minus_1 + torch.diag(w) @ Z_t @ torch.diag(w)
            trace_term = torch.trace(A_t)
            loss = F.softplus(a - f) + v * trace_term
        else:  # y == 0
            A_t = A_t_minus_1 - torch.diag(w) @ Z_t @ torch.diag(w)
            trace_term = torch.trace(A_t)
            loss = f + v * trace_term

        return loss

    
    def optimize_w_torch_softplus(self, u, A_t_minus_1, Z_t, a, v, y, lr=1e-2, steps=500):
        d = u.shape[0]
        w = torch.ones(d, requires_grad=True)

        optimizer = torch.optim.Adam([w], lr=lr)
        u_tens = torch.FloatTensor(u)
        A_t_minus_1_tens = torch.FloatTensor(A_t_minus_1)
        Z_t_tens = torch.FloatTensor(Z_t)
        with torch.no_grad():
            #first_loss = self.softplus_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v).item()
            first_loss = self.piecewise_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v, y=y).item()
            print("First loss: ", first_loss)
        for step in range(steps):
            optimizer.zero_grad()
            #loss = self.softplus_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v)
            loss = self.piecewise_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v, y=y)
            loss.backward()
            optimizer.step()
            if y == 0:
                with torch.no_grad():
                    w.data = self.project_w(w.data, Z_t=Z_t_tens, A_prev=A_t_minus_1_tens)

            

            # if step % 50 == 0 or step == steps - 1:
            #     print(f"Step {step}: Loss = {loss.item():.6f}")
        with torch.no_grad():
            #final_loss = self.softplus_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v).item()
            final_loss = self.piecewise_loss(w=w, u=u_tens, A_t_minus_1=A_t_minus_1_tens, Z_t=Z_t_tens, a=a, v=v, y=y).item()
            print("Final loss: ", final_loss)
            print("Loss Decreased: ", first_loss > final_loss)
        return w.detach().numpy()
    
    def project_w(self, w, Z_t, A_prev):
        """
        Project w so that A_prev - diag(w) z z^T diag(w) is PSD
        """
        lambda_min = torch.linalg.eigvalsh(A_prev).min().item()
        z = self.compute_z_from_Z(Z_t)
        Dz = w * z
        norm_sq = torch.dot(Dz, Dz)

        if norm_sq > lambda_min:
            scale = (lambda_min / (norm_sq + 1e-12))**0.5
            w = w * scale
        return w
    
    def compute_z_from_Z(self, Z_t):
        # Eigen decomposition (Z_t = z z^T)
        eigvals, eigvecs = torch.linalg.eigh(Z_t)
        
        # Find the leading eigenvector (only one non-zero since rank-1)
        idx = torch.argmax(eigvals)
        z = eigvecs[:, idx]

        # Ensure unit norm (safety)
        z = z / torch.norm(z)

        return z







    
    def deweight_features(self, u_point, mean):
        diff = u_point - mean
        mahal_grad = self.curr_dist_mat @ diff
        contributions = np.abs(diff*mahal_grad)
        # get the top k contributions indices
        if self.args.full_k:
            print("> Using Full K Deweighting... ")
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


@njit
def optimize_w_psd_projected_softplus_batch(x_i,
        x_list, mu_X, A_t_minus1, Z_t, v, p, y_i, y_list, a=2.0,
        lr=1e-2, steps=100
    ):
    if y_i == 0:
        w = np.random.randn(p) * 0.01
    else:
        w = np.ones(p)
    # Compute D_z = diag(z) (where z is the eigenvector of Z_t)
    z = np.linalg.svd(Z_t)[0][:, 0]  # Take the first eigenvector of Z_t
    D_z = np.diag(z)

    first_loss = None
    first_dev = None
    for j in range(steps):
        curr_grad_loss = np.zeros_like(w)
        curr_loss = 0
        for i in range(len(x_list)):
            x_curr = x_list[i]
            y_curr = y_list[i]
            u = x_curr - mu_X
            d0 = u @ A_t_minus1 @ u
            Q = np.diag(u) @ Z_t @ np.diag(u)
            quad = w @ Q @ w
            trace_quad = w @ Z_t @ w

            if y_i == 0:
                d = d0 - quad
                trace_term = -trace_quad
            else:
                d = d0 + quad
                trace_term = +trace_quad

            dev = d
            trace_val = np.trace(A_t_minus1) + trace_term

            # Loss calculation (no penalty for the PSD constraint)
            if y_curr == 0:
                loss = abs(dev) + v * trace_val
            else:
                #loss = max(0.0, a - dev) + v * trace_val
                loss = np.log1p(np.exp(a - dev)) + v * trace_val

            # Gradient of dev
            if y_i == 0:
                grad_dev = -2 * Q @ w
                grad_trace = -2 * Z_t @ w
            else:
                grad_dev = +2 * Q @ w
                grad_trace = +2 * Z_t @ w

            # Final gradient
            if y_curr == 0:
                grad_loss = np.sign(dev) * grad_dev + v * grad_trace
            else:
                # negative sigmoid for softplus
                grad_loss = -np.exp(a - dev)/(1 + np.exp(a - dev)) * grad_dev + v * grad_trace
            curr_grad_loss += grad_loss
            curr_loss += loss

        # Gradient step
        w -= lr * curr_grad_loss

        # Projection to maintain the PSD constraint (for y = 0)
        # Compute M = D_z A_{t-1}^{-1} D_z
        A_inv = np.linalg.inv(A_t_minus1)
        M = D_z @ A_inv @ D_z

        # Check if the constraint is violated
        constraint_value = w @ M @ w
        if constraint_value > 1.0:
            # Scale w to satisfy w^T M w = 1
            scale = np.sqrt(1.0 / constraint_value)
            w *= scale
        if j == 0:
            first_loss = curr_loss
    print("First loss: ", first_loss)
    print("Final loss: ", curr_loss)
    print("Loss Decreased: ", first_loss > curr_loss)
    return w



    