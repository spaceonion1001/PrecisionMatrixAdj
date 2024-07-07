import numpy as np 
import pandas as pd
import os

from numba import njit
from sklearn.preprocessing import LabelEncoder

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

def load_unsw_data(args):
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