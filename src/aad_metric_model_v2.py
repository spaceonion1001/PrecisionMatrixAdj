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

        self.a = 3 # significance level
    
    def iterate(self):
        w_s = []
        Z_s = []

        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.num_preds = []

        for j in range(self.J):
            print(">>> ITER {} <<<".format(j))

            delta, curr_Z, is_anomaly, mu, u_point = self.get_user_feedback(args=self.args)
            if is_anomaly:
                w, A_t = self.optimize_w(delta=delta, Z_t=curr_Z, A_prev=self.curr_dist_mat, y=1, a=self.a, v=self.v, lr=1e-2, max_iters=100, tol=1e-5, verbose=True)
                self.w_s.append(w)
                self.Z_s.append(curr_Z)
                self.curr_dist_mat = A_t
            else:
                self.deweight_features(u_point=u_point, mean=mu)
                

            self.num_anomalies_queried_list.append(self.num_anomalies_queried)

        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        np.savetxt('figures/debugging_figs_true/num_anomalies_queried_{}_{}.csv'.format(self.args.data, self.args.save_suffix), np.array(self.num_anomalies_queried_list).astype(int))

        return self.curr_dist_mat
            


    def get_user_feedback(self, args):
        use_data = self.data[self.masked_data, :]
        use_labels = self.labels[self.masked_data]
        print("Use Data Shape: ", use_data.shape)
        mu = np.mean(self.original_data, axis=0)
        anomaly_scores = calc_anomaly_scores(features=use_data, A_t=self.curr_dist_mat, mean_vec=mu)


        # greedy selection
        selected_idx = np.argmax(anomaly_scores)

        selected_score = anomaly_scores[selected_idx]
        selected_point = use_data[selected_idx, :]
        selected_label = use_labels[selected_idx]
        print("Selected Point Score: {}".format(selected_score))
        print("Score Mean: {}".format(np.mean(anomaly_scores)))
        print("Score Max: {}".format(np.max(anomaly_scores)))
        print("Score Min: {}".format(np.min(anomaly_scores)))

        diff = selected_point - mu
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

        return loss, grad, A_t

    def optimize_w(self, delta, Z_t, A_prev, y, a, v, lr=1e-2, max_iters=100, tol=1e-6, verbose=False):
        """
        Optimize w using gradient descent.
        """
        d = delta.shape[0]
        w = np.zeros(d)
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
            self.curr_dist_mat = w_precision_update(self.curr_dist_mat, top_k_idx, direction_vector=diff)





def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis_fast_uv(v, mean_vec, A_t)
        dists.append(curr_dist)

    return np.array(dists)



    