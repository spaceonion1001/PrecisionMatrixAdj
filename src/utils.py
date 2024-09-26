import numpy as np 
import pandas as pd
import os

from numba import njit
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
np.random.seed(42)

@njit
def is_pos_def(A):
    if is_symmetric(A):
        try:
            np.linalg.cholesky(A)
            return True
        #except np.linalg.LinAlgError:
        except:
            return False
    else:
        return False
@njit
def is_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

#@njit
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval <= 0.0] = 1e-8

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def load_census_data(args):
    path = args.data_path

    census_path = os.path.join(path, 'census_1.csv')
    data = pd.read_csv(census_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_wine_data(args):
    path = args.data_path

    wine_path = os.path.join(path, 'wine.data')
    data = pd.read_csv(wine_path, header=None)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    ss = StandardScaler()
    features = ss.fit_transform(features)

    return data.values, features.astype(float), classes

def load_unsw_data(args):
    print("*** LOADING UNSW DATA ***")
    path = args.data_path

    unsw_path = os.path.join(path, 'unsw_1.csv')
    data = pd.read_csv(unsw_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))
    

    return data.values, features.astype(float), classes

def load_kddcup_data(args):
    path = args.data_path

    kddcup_path = os.path.join(path, 'kddcup_1.csv')
    data = pd.read_csv(kddcup_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_campaign_data(args):
    path = args.data_path

    campaign_path = os.path.join(path, 'campaign_1.csv')
    data = pd.read_csv(campaign_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_kdd_donors_data(args):
    path = args.data_path

    kdd_path = os.path.join(path, 'KDD2014_donors_10feat_nomissing_normalised.csv')
    data = pd.read_csv(kdd_path)
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes


def load_bank_data(args):
    path = args.data_path

    bank_path = os.path.join(path, 'bank-additional-full_normalised.csv')
    data = pd.read_csv(bank_path)
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes

def load_retinopathy_data(args):
    path = args.data_path

    retinopathy_path = os.path.join(path, 'retinopathy.arff')
    data = arff.loadarff(retinopathy_path)
    data = pd.DataFrame(data[0]).iloc[:, 1:]
    data['Class'] = data['Class'].str.decode('utf-8').astype('int')
    
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes

def load_cancer_data(args):
    path = args.data_path

    cancer_path = os.path.join(path, 'breast_cancer.DATA')
    data = pd.read_csv(cancer_path, header=None).iloc[:, 1:]
    features = data.values[:, 1:].astype(float)
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))
    classes = 1 - classes

    return data.values, features, classes


def add_epsilon_noise(features):
    print("Adding Epsilon Noise {} ...".format(features.shape))
    for j in range(features.shape[1]):
        features[:, j] += np.random.normal(scale=1e-6, size=features.shape[0])
    
    return features

def remove_bad_features(features):
    print("Removing degenerate features {}".format(features.shape))
    stds = np.std(features, axis=0)
    features = features[:, stds != 0.0]
    print("New shape {}".format(features.shape))
    
    return features

def smart_sampling(features, labels, num_anoms=10, num_nominals=100):
    print("Sampling data - margin based (Manhattan Distance)")

    # calculate Manhattan distance between all points and the mean
    feat_mean = np.mean(features, axis=0)
    dist_vec = np.zeros((features.shape[0],))

    for i in range(features.shape[0]):
        curr_dist = np.sum(np.abs(features[i] - feat_mean))
        dist_vec[i] = curr_dist
    
    # get indices of labels with anomaly class (designated class 0)
    anom_indices = np.asarray(labels == 0).nonzero()[0]
    nominal_indices = np.asarray(labels != 0).nonzero()[0]


    # need to get top K or whatever indices of nominals close to mean and far from mean
    # get anomalies close to mean
    # build triplets of these
    dist_vec_anoms = dist_vec[anom_indices]
    dist_vec_nominals = dist_vec[nominal_indices]
    features_anoms = features[anom_indices, :]
    features_nominals = features[nominal_indices, :]

    anoms_sorted = np.argsort(dist_vec_anoms)[::-1]
    nominals_sorted = np.argsort(dist_vec_nominals)[::-1]

    keep_anomalies_idx = anoms_sorted[-num_anoms:]
    keep_nominals_far_idx = nominals_sorted[0:num_nominals]
    keep_nominals_close_idx = nominals_sorted[-num_nominals:]
    keep_nominals_total_idx = np.array(list(set(np.append(keep_nominals_far_idx, keep_nominals_close_idx))))
    
    anomaly_pts = features_anoms[keep_anomalies_idx, :]
    nominal_pts = features_nominals[keep_nominals_total_idx, :]
    anom_labels = np.zeros(anomaly_pts.shape[0])
    nom_labels = np.ones(nominal_pts.shape[0])
    total_feats = np.concatenate((anomaly_pts, nominal_pts), 0)
    total_labels = np.concatenate((anom_labels, nom_labels), 0)

    return total_feats, total_labels

    
