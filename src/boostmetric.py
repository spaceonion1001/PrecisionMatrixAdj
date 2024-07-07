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


from gmm import sim_gmm


class BoostMetric:
    def __init__(self, data, labels, v, J):
        self.data = data
        self.labels = labels
        self.v = v
        self.J = J
        self.triplet_idxs = convert_to_triples(data, labels)
        print("Number of Triplets {}".format(len(self.triplet_idxs)))
        self.A_arr = calculate_AR(self.data, self.triplet_idxs)
        print("A length {}".format(self.A_arr.shape))

        self.u_r = np.ones(self.A_arr.shape[0])*(1/self.A_arr.shape[0])

    def iterate(self):

        for j in range(self.J):
            A_hat_curr = self.calc_Ahat()
            #print("A_hat", A_hat_curr)
            eigvals, eigvecs = np.linalg.eig(A_hat_curr)
            #print(eigvals, eigvecs)
            print(eigvals)
            break
        

    def calc_Ahat(self):
        A_hat = np.zeros((self.A_arr.shape[1], self.A_arr.shape[2]))

        for i in range(self.A_arr.shape[0]):
            A_hat = A_hat + self.u_r[i]*self.A_arr[i]

        # if not is_pos_def(A_hat):
        #     print("Adjusting")
        #     A_hat = get_near_psd(A_hat)

        return A_hat


@njit
def convert_to_triples(data, labels):
    triplet_idxs = []
    for i in range(len(labels)-2):
        for j in range(i+1, len(labels)-1):
            for k in range(i+2, len(labels)):
                curr_i = labels[i]
                curr_j = labels[j]
                curr_k = labels[k]
                if curr_i == curr_j and curr_i != curr_k:
                    triplet_idxs.append((i, j, k))
    
    return triplet_idxs

#@njit
def calculate_AR(data, triplet_idxs):
    A_arr = []
    for tup in triplet_idxs:
        a_i = np.expand_dims(data[tup[0]], 1)
        a_j = np.expand_dims(data[tup[1]], 1)
        a_k = np.expand_dims(data[tup[2]], 1)
        ai_m_ak = (a_i-a_k)
        ai_m_aj = (a_i-a_j)
        first = ai_m_ak@ai_m_ak.T #ai_m_ak.dot(ai_m_ak.T)
        second = ai_m_aj@ai_m_aj.T #ai_m_aj.dot(ai_m_aj.T)
        
        #first = first + 10*np.eye(first.shape[0])
        #second = second + 10*np.eye(first.shape[0])

        A_r = first - second
        # A_r = A_r + 10*np.eye(A_r.shape[0])
        # if not is_pos_def(A_r):
        #     print(a_i)
        #     print(a_j)
        #     print(a_k)
        #     print("First", first)
        #     print("Second", second)
        #     print("Ai-Ak", ai_m_ak)
        #     print("Ai-Aj", ai_m_aj)
        #     print("A_r", A_r)
        #     print("Diag Total ", np.diag(A_r))
        #     print("Eig Total ", np.linalg.eig(A_r)[0])
        #     print("Diag First ", np.diag(first))
        #     print("Eig First ", np.linalg.eig(first)[0])
        #     print("Diag Second ", np.diag(second))
        #     print("Eig Second ", np.linalg.eig(second)[0])
        #     exit()
        # if not is_pos_def(A_r):
        #     A_r = get_near_psd(A_r)
        A_arr.append(A_r)

    return np.array(A_arr)

data, labels = sim_gmm(N=18, dim=10)
print(data.shape, labels.shape)

bm = BoostMetric(data=data, labels=labels, v=10e-7, J=100)
bm.iterate()