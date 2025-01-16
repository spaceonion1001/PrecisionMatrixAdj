import numpy as np
from sklearn.utils.extmath import fast_logdet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from numpy.linalg import inv

from numba import njit
from numba_progress import ProgressBar

from utils import *
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from autoencoder import AE, train_model
from torch.utils.data import TensorDataset, DataLoader
import torch
from statsmodels.stats.moment_helpers import cov2corr
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='C:\\Users\\xdink\\Documents\\Research\\OnlineMetricLearning\\data')
    parser.add_argument('--data', type=str, default='census')
    parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--inter_update', action='store_true')
        

    args = parser.parse_args()

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
    else:
        print("Incorrect Dataset...")
        exit(1)

#@njit
def true_y(ulabel, vlabel):
    # simple binary function
    if ulabel == vlabel:
        print("SAME LABEL")
        return 0.0
    else:
        print("DIFFERENT LABEL")
        return 10.0
    
#@njit
def calc_ybar(eta, y_t, yhat_t):
    denom = 2*eta*yhat_t

    first = eta*y_t*yhat_t - 1
    inn = (eta*y_t*yhat_t - 1)**2 + 4*eta*(yhat_t)**2
    numer = first + np.sqrt(inn)
    if denom <= 0.0:
        denom = 1e-6

    return numer/denom

#@njit
def calc_A_tpo(eta, ybar, y_t, A_t, u_t, v_t):
    z_t = u_t - v_t
    first = eta*(ybar - y_t)
    second = A_t@z_t@z_t.T@A_t
    numer = first*second

    dsecond = z_t.T@A_t@z_t
    denom = 1 + first*dsecond

    A_tpo = A_t - numer/denom
    
    if not is_pos_def(A_tpo):
        print("** FIXING PSD **")
        A_tpo = get_near_psd(A_tpo)
    
    assert is_pos_def(A_tpo)
    assert is_symmetric(A_tpo)

    return A_tpo

def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis(mean_vec, v, A_t)
        dists.append(curr_dist)

    return np.array(dists)

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

class OML:
    def __init__(self, budget, eta, data, labels, init_dist_mat, args):
        self.budget = budget
        self.eta = eta
        self.data = data.copy()
        self.labels = labels.copy()
        self.init_dist_mat = init_dist_mat
        self.queried_labels = []
        self.queried_points = []
        self.args = args
        self.orig_mu = np.mean(self.data, axis=0)

    def iterate(self):
        self.A_t = self.init_dist_mat
        # v_t is mean of data
        v_t = self.data.mean(axis=0)
        v_t = np.expand_dims(v_t, 1)
        v_label = 1 # always nominal point
        for j in tqdm(range(self.budget)):
            curr_anomaly_scores = calc_anomaly_scores(self.data, self.A_t, self.orig_mu)
            # get highest anomaly score
            max_idx = np.argmax(curr_anomaly_scores)
            u_t = self.data[max_idx, :]
            u_label = self.labels[max_idx]

            # remove from data
            self.data = np.delete(self.data, max_idx, axis=0)
            self.labels = np.delete(self.labels, max_idx, axis=0)

            u_t = np.expand_dims(u_t, 1)
            y_t = true_y(u_label, v_label)
            yhat_t = mahalanobis(u_t.flatten(), v_t.flatten(), self.A_t)
            ybar = calc_ybar(eta=self.eta, y_t=y_t, yhat_t=yhat_t)
            A_tpo = calc_A_tpo(eta=self.eta, ybar=ybar, y_t=y_t, A_t=self.A_t, u_t=u_t, v_t=v_t)
            self.A_t = A_tpo
            if self.args.inter_update:
                if len(self.queried_points) > 0:
                    self.update_query_set(u_t, u_label)
            self.queried_points.append(u_t)
            self.queried_labels.append(u_label)
        
        return self.A_t
    
    def update_query_set(self, u_t, u_label):
        print("Performing Inter-Update")
        for i in range(len(self.queried_points)):
            v_t = self.queried_points[i]
            #v_t = np.expand_dims(v_t, 1)
            v_label = self.queried_labels[i]
            y_t = true_y(u_label, v_label)
            yhat_t = mahalanobis(u_t.flatten(), v_t.flatten(), self.A_t)
            ybar = calc_ybar(eta=self.eta, y_t=y_t, yhat_t=yhat_t)
            A_tpo = calc_A_tpo(eta=self.eta, ybar=ybar, y_t=y_t, A_t=self.A_t, u_t=u_t, v_t=v_t)
            self.A_t = A_tpo

def predict_percentile(args, features, X, labels, percentile=95, dist='mahalanobis'):
    mu = np.mean(features, axis=0)
    dists = []
    for i in range(features.shape[0]):
        a_i = features[i, :]
        if dist == 'mahalanobis' or dist == 'mahalanobis_default' or dist == 'mahalanobis_default_eye':
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
    plt.savefig('./figures/debugging_figs_oml/dist_hist_{}_{}_{}'.format("FINAL", dist, args.data))
    plt.close()
    return preds

def main(args):
    data, features, labels = resolve_data(args)
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
    features = remove_bad_features(features)
    scaler = StandardScaler()
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
    init_dist_mat = init_covar(features, normalize=args.normalize, identity=args.identity)

    oml = OML(budget=args.budget, eta=args.eta, data=features, labels=labels, init_dist_mat=init_dist_mat, args=args)
    X = oml.iterate()
    preds = predict_percentile(args, features, X, labels, percentile=95, dist='mahalanobis')
    print("Mahal Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds), precision_score(1-labels, 1-preds), recall_score(1-labels, 1-preds)))
    preds_mahal_default_percentile = predict_percentile(args, features, X=inv(np.cov(features.T)), labels=labels, percentile=95, dist='mahalanobis_default')
    print("Mahal Default Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_default_percentile), precision_score(1-labels, 1-preds_mahal_default_percentile), recall_score(1-labels, 1-preds_mahal_default_percentile)))
    preds_mahal_default_eye = predict_percentile(args, features, X=inv(np.eye(features.shape[1])), labels=labels, percentile=95, dist='mahalanobis_default_eye')
    print("Mahal Default Eye: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_default_eye), precision_score(1-labels, 1-preds_mahal_default_eye), recall_score(1-labels, 1-preds_mahal_default_eye)))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)