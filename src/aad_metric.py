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
from sklearn.svm import SVC
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score
from sklearn.utils import shuffle
from numpy.linalg import inv

from numba import njit
from numba_progress import ProgressBar

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

class BoostMetric:
    def __init__(self, data, labels, v, J, init_dist_mat, args, keep_top_k=False, top_k=3):
        self.data = data
        self.labels = labels
        self.v = v
        self.J = J
        # self.triplet_idxs = convert_to_triples(data, labels)
        # print("Number of Triplets {}".format(len(self.triplet_idxs)))
        #self.A_arr = calculate_AR(self.data, self.triplet_idxs)
        #print("A length {}".format(self.A_arr.shape))

        #self.u_r = np.ones(self.A_arr.shape[0])*(1/self.A_arr.shape[0])
        self.u_r_arr = []
        self.u_r = None
        self.triplets = []
        self.curr_dist_mat = init_dist_mat
        # self.keep_top_k = keep_top_k
        self.top_k = top_k
        self.args = args

    def iterate(self):
        w_s = []
        Z_s = []
        margins_count = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        for j in range(self.J):
            # A_hat_curr = self.calc_Ahat()
            # if not is_symmetric(A_hat_curr):
            #     A_hat_curr = (A_hat_curr + A_hat_curr.T)/2
            # #print("A_hat", A_hat_curr)
            # eigvals, eigvecs = np.linalg.eigh(A_hat_curr)
            # max_ind = np.argmax(eigvals)
            # max_eig = eigvals[max_ind]
            # max_vec = np.expand_dims(eigvecs[max_ind], 1)
            # curr_Z = max_vec @ max_vec.T

            # USER FEEDBACK
            dists, curr_Z, curr_triplet = self.replicate_user_feedback(args=self.args, itr=j)
            met_count = self.calc_sigma_metric(dists=dists)
            if not is_symmetric(curr_Z):
                curr_Z = (curr_Z + curr_Z.T)/2
            self.triplets.append(curr_triplet)
            self.u_r_arr.append(1.0/(len(self.u_r_arr)+1))

            # normalize u
            self.u_r = np.array(self.u_r_arr)
            self.u_r = self.u_r/self.u_r.sum()
            self.u_r_arr = self.u_r.tolist()
            
            # calculate A_r for all current triplets
            A_arr = calculate_AR(self.triplets)

            # calculate H_s (margins) for current user feedback
            H_s = self.calc_H_r_js(curr_Z=curr_Z, A_arr=A_arr)

            # calculate weight with bisection search
            w_j = self.bisection_search(curr_Z, H_s)

            print("W_j {}".format(w_j))
            sns.heatmap(curr_Z)
            plt.savefig('figures/debugging_figs/Z_{}_{}.png'.format(self.args.data, j))
            plt.close()          
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # update u
            # TODO
            # FIX THIS SECTION
            new_u = []
            for r in range(self.u_r.shape[0]):
                u_r_new = self.u_r[r]*np.exp(-H_s[r]*w_j)
                #u_r_new = np.exp(-H_s[r]*w_j)
                print("U{} {}".format(r, u_r_new))
                new_u.append(u_r_new)
            new_u = np.array(new_u)
            new_u = new_u / new_u.sum()
            self.u_r = new_u
            self.u_r_arr = self.u_r.tolist()
            print(self.u_r)
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # ------------------------------------------------------------ #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


            w_s.append(w_j)
            #w_s.append(1.0)
            Z_s.append(curr_Z)
            #print("W_j", w_j)
            #print("Z Shape", curr_Z.shape)
            # exclusion of sigma inv
            # if j == 0:
            #     self.curr_dist_mat = np.array(w_s)[0]*np.array(Z_s)[0]
            # else:
            #     for i in range(len(w_s)):
            #         self.curr_dist_mat += np.array(w_s)[i]*np.array(Z_s)[i]

            # no exclusion of sigma inv
            for i in range(len(w_s)):
                self.curr_dist_mat += np.array(w_s)[i]*np.array(Z_s)[i]

            # margin calculations
            print("\nCurrent Margins...")
            for i in range(len(A_arr)):
                H_i = np.trace(A_arr[i] @ self.curr_dist_mat.T)
                print("   >",H_i)

            sns.heatmap(self.curr_dist_mat)
            plt.savefig('figures/debugging_figs/X_{}_{}.png'.format(self.args.data, j))
            plt.close()
            print()
            curr_f1, curr_precision, curr_recall = final_classification(features=self.data, labels=self.labels, X=self.curr_dist_mat)
            self.f1s.append(curr_f1)
            self.precisions.append(curr_precision)
            self.recalls.append(curr_recall)

        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        
        return self.curr_dist_mat
        
    def replicate_user_feedback(self, args, itr=0):
        dists = []
        mu = np.mean(self.data, axis=0)
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, mu, self.curr_dist_mat)
            dists.append(curr_dist)
        dists = np.array(dists)
        anom_dists = dists[self.labels == 0]
        nominal_dists = dists[self.labels == 1]

        # save histogram of distances
        sns.histplot(anom_dists, alpha=0.5, label='anomalies', kde=True)
        sns.histplot(nominal_dists, alpha=0.5, label='nominal', kde=True)
        plt.legend(loc='best')
        plt.xlabel('Distance to Mean')
        plt.tight_layout()
        plt.savefig('./figures/debugging_figs/dist_hist_{}_{}'.format(args.data, itr))
        plt.close()

        anom_feats = self.data[self.labels == 0, :]
        anom_dists_mean = np.mean(anom_dists)
        #print("USING ANOM DIST MAX")
        #anom_dists_mean = np.max(anom_dists)
        # anomaly specific index
        a_k_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
        selected_anom_idx = np.where(self.labels==0)[0][a_k_indx]
        #a_k_indx = np.argmin(anom_dists)
        a_k = anom_feats[a_k_indx, :]
        # get distance from all points to a_k
        a_k_dists = []
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, a_k, self.curr_dist_mat)
            a_k_dists.append(curr_dist)
        a_k_dists = np.array(a_k_dists)
        nom_dists_ak = a_k_dists[self.labels==1]
        
        # keep just the top 1000, or all if it's less than that
        num_nominal_keep = min(anom_feats.shape[0], nom_dists_ak.shape[0])
        top_k_indx = np.argpartition(nom_dists_ak, -num_nominal_keep)[-num_nominal_keep:]
        D_nom = self.data[self.labels==1, :][top_k_indx, :]
        D_tot = np.concatenate((anom_feats, D_nom), axis=0)
        anom_labels = np.zeros(anom_feats.shape[0])
        nom_labels = np.ones(D_nom.shape[0])
        label_tot = np.concatenate((anom_labels, nom_labels))

        # fit decision boundary
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(D_tot, label_tot)
        print("Classification Training Score {}".format(clf.score(D_tot, label_tot)))

        # calculate top_k most important features
        coef_abs = np.abs(clf.coef_)
        coef_topk_idx = np.argpartition(coef_abs[0], -self.top_k)[-self.top_k:]

        # construct weak learner from vector of most important features
        coef_copy = clf.coef_.copy()[0]
        z_vec = np.zeros_like(coef_copy)
        z_vec[coef_topk_idx] = coef_copy[coef_topk_idx]
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        curr_Z = z_vec@z_vec.T

        # construct triplet
        # THIS COULD BE TOTAL NUMBER OF NOMINALS NOT JUST D-NOM
        w_norm = np.linalg.norm(clf.coef_)
        dnom_dists = clf.decision_function(D_nom)/w_norm
        preds = clf.predict(D_nom)
        min_dnom_dists = dnom_dists[preds == 1] # just the nominal prediction side of decision boundary
        st_idx = np.argpartition(min_dnom_dists, 2)[:2]
        a_1_idx = np.where(preds == 1)[0][st_idx[0]]
        a_2_idx = np.where(preds == 1)[0][st_idx[1]]
        a_1 = D_nom[a_1_idx]
        a_2 = D_nom[a_2_idx]
        curr_triplet = (a_1, a_2, a_k)
        selected_a1_idx = np.where(self.labels == 1)[0][st_idx[0]]
        selected_a2_idx = np.where(self.labels == 1)[0][st_idx[1]]
        print("Dist a1->a2", mahalanobis(a_1, a_2, self.curr_dist_mat))
        print("Dist a1->ak", mahalanobis(a_1, a_k, self.curr_dist_mat))
        print("Dist a2->ak", mahalanobis(a_2, a_k, self.curr_dist_mat))

        self.data = np.delete(self.data, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0)
        self.labels = np.delete(self.labels, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0)
        
        return np.delete(dists, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0), curr_Z, curr_triplet


    def calc_Ahat(self):
        A_hat = np.zeros((self.A_arr.shape[1], self.A_arr.shape[2]))

        for i in range(self.A_arr.shape[0]):
            A_hat = A_hat + self.u_r[i]*self.A_arr[i]

        # if not is_pos_def(A_hat):
        #     print("Adjusting")
        #     A_hat = get_near_psd(A_hat)

        return A_hat
    
    def bisection_search(self, curr_Z, H_s, w_l=1e-5, w_u=1.0, eps=1e-4):
        w_j = 0.0
        while w_u - w_l >= eps:
            w_j = 0.5*(w_u + w_l)
            lhs = self.calc_lhs(H_s, w_j)
            #print(lhs)
            if lhs > 0:
                w_l = w_j
            else:
                w_u = w_j
        
        return w_j
            

    def calc_H_r_js(self, curr_Z, A_arr):
        H_s = []
        for r in range(A_arr.shape[0]):
            #print("A", self.A_arr[r].shape)
            #print("Z", curr_Z.shape)
            H_r_j = np.trace(A_arr[r] @ curr_Z.T)
            #print("H", H_r_j.shape)
            
            H_s.append(H_r_j)

        return np.array(H_s)
    
    def calc_lhs(self, H_s, w_j):
        summand = 0
        for r in range(len(self.u_r)):
            first = (H_s[r] - self.v)
            second = self.u_r[r]
            third = np.exp(-w_j*H_s[r]) # overflow can happen here...
            summand += first*second*third
        return summand
    
    def get_w_Z(self):
        return self.w_s, self.Z_s
    
    def calc_sigma_metric(self, dists):
        three_sigma_empirical = 3*dists.var()
        # num_points_outside = (dists > three_sigma_empirical).sum()
        top_40_far_idx = np.argpartition(dists, -40)[-40:]
        top_40_far_labels = self.labels[top_40_far_idx]
        top_40_close_idx = np.argpartition(dists, 40)[:40]
        top_40_close_labels = self.labels[top_40_close_idx]
        print("Farthest 40 Points Anomalies {}".format((top_40_far_labels == 0).sum()))
        print("Closest 40 Points Anomalies {}".format((top_40_close_labels == 0).sum()))
        print("Three Sigma {}".format(three_sigma_empirical))
        print("Dist Mean {} Min {} Max {}".format(dists.mean(), dists.min(), dists.max()))

        return (top_40_far_labels == 0).sum()
    


#@njit
def convert_to_triples(data, labels):
    #nn = NearestNeighbors(n_neighbors=)
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
def calculate_AR(triplets):
    A_arr = []
    for tup in triplets:
        a_i = np.expand_dims(tup[0], 1)
        a_j = np.expand_dims(tup[1], 1)
        a_k = np.expand_dims(tup[2], 1)
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
        A_r = (A_r + A_r.T)/2
        assert is_symmetric(A_r)
        A_arr.append(A_r)
        
    return np.array(A_arr)

def init_covar(data):
    A_t = np.cov(data.T)
    #A_t = np.corrcoef(data.T)
    A_t = A_t + 1e-5*np.eye(A_t.shape[0])

    assert is_pos_def(A_t)
    return inv(A_t)

def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis(mean_vec, v, A_t)
        dists.append(curr_dist)

    return np.array(dists)

def mahal_all_points(features, X):
    dist_mat = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            u = features[i, :]
            v = features[j, :]
            curr_dist = mahalanobis(u, v, X)
            dist_mat[i, j] = curr_dist
    
    return np.array(dist_mat)

def final_classification(features, labels, X):
    dists = []
    chisq_vals = []
    mu = np.mean(features, axis=0)
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = mahalanobis(a_i, mu, X)
        dists.append(curr_dist)
        chisq_vals.append(chi2.cdf(curr_dist, df=features.shape[1]-1))
    dists = np.array(dists)
    chisq_vals = np.array(chisq_vals)
    sigma = dists.std()
    preds_chisq = np.zeros_like(labels)
    preds_chisq[chisq_vals <= 0.1] = 1
    #print("ChiSq: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(labels, preds_chisq), f1_score(labels, preds_chisq), precision_score(labels, preds_chisq), recall_score(labels, preds_chisq)))
    #print("Avg Precision Score {}".format(average_precision_score(labels, chisq_vals)))

    # class zero is anomaly, switch it for sake of calculation
    return f1_score(1-labels, preds_chisq), precision_score(1-labels, preds_chisq), recall_score(1-labels, preds_chisq)
    


if __name__ == "__main__":
    args = parse_arguments()
    #main(args)
    #main_boost(args)

    data, features, labels = resolve_data(args)
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
    else:
        labels[labels != 0] = 1

    features = scaler.fit_transform(features)
    init_dist_mat = init_covar(features)
    print("Data {} Shape {}".format(args.data, features.shape))

    bm = BoostMetric(data=features, labels=labels, init_dist_mat=init_dist_mat, args=args, v=args.v, J=args.iters, top_k=args.k)
    X = bm.iterate()
    plt.plot(np.arange(args.iters), bm.f1s, label="F1")
    plt.plot(np.arange(args.iters), bm.precisions, label='Precision')
    plt.plot(np.arange(args.iters), bm.recalls, label='Recall')
    plt.xlabel("Iterations")
    plt.ylabel("Scores")
    plt.legend(loc='best')
    plt.title("Scores {}".format(args.data))
    plt.savefig('figures/debugging_figs/scores_{}'.format(args.data))
    plt.close()
    #final_classification(features=features, labels=labels, X=X)