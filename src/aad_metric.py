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
    else:
        print("Incorrect Dataset...")
        exit(1)

class BoostMetric:
    def __init__(self, data, labels, v, J, init_dist_mat, args, keep_top_k=False, top_k=3):
        self.data = data
        self.original_data = data.copy()
        self.labels = labels
        self.original_labels = labels.copy()
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
        self.init_dist_mat = init_dist_mat

    def iterate(self):
        w_s = []
        Z_s = []
        margins_count = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.num_preds = []
        for j in range(self.J):
            print(">>> ITER {} <<<".format(j))
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
            if self.args.double:
                dists, curr_Z_anom, curr_Z_nom, curr_triplet, second_triplet = self.replicate_user_feedback_double(args=self.args, itr=j, closer=self.args.closer, negate=self.args.negate)
                if not is_symmetric(curr_Z_anom):
                    curr_Z_anom = (curr_Z_anom + curr_Z_anom.T)/2
                if not is_symmetric(curr_Z_nom):
                    curr_Z_nom = (curr_Z_nom + curr_Z_nom.T)/2
                self.triplets.append(curr_triplet)
                self.triplets.append(second_triplet)
                if len(self.u_r_arr) >= 1:
                    # append twice for two triplets
                    self.u_r_arr.append(np.array(self.u_r_arr).sum()/(len(self.u_r_arr)+1))
                    self.u_r_arr.append(np.array(self.u_r_arr).sum()/(len(self.u_r_arr)+1))
                else:
                    # append twice for two triplets
                    self.u_r_arr.append(1.0)
                    self.u_r_arr.append(1.0)
                # normalize u
                self.u_r = np.array(self.u_r_arr)
                self.u_r = self.u_r/self.u_r.sum()
                self.u_r_arr = self.u_r.tolist()
                A_arr = calculate_AR(self.triplets)
                H_s_anom = self.calc_H_r_js(curr_Z=curr_Z_anom, A_arr=A_arr)
                H_s_nom = self.calc_H_r_js(curr_Z=curr_Z_nom, A_arr=A_arr)
                w_j_anom = self.bisection_search(curr_Z_anom, H_s_anom)
                w_j_nom = self.bisection_search(curr_Z_nom, H_s_nom)
                if self.args.negate:
                    w_j_nom = self.bisection_search_posdef(curr_Z=curr_Z_nom, curr_dist_mat=self.curr_dist_mat, w_l=0.0, w_u=w_j_nom)
                # update u twice for two triplets
                new_u = []
                for r in range(self.u_r.shape[0]):
                    u_r_new = self.u_r[r]*np.exp(-H_s_anom[r]*w_j_anom)
                    new_u.append(u_r_new)
                new_u = np.array(new_u)
                new_u = new_u / new_u.sum()
                self.u_r = new_u
                self.u_r_arr = self.u_r.tolist()
                new_u = []
                for r in range(self.u_r.shape[0]):
                    u_r_new = self.u_r[r]*np.exp(-H_s_nom[r]*w_j_nom)
                    new_u.append(u_r_new)
                new_u = np.array(new_u)
                new_u = new_u / new_u.sum()
                self.u_r = new_u
                self.u_r_arr = self.u_r.tolist()
                w_s.append(w_j_anom)
                w_s.append(w_j_nom)
                Z_s.append(curr_Z_anom)
                Z_s.append(curr_Z_nom)
                self.curr_dist_mat += w_j_anom*curr_Z_anom 
                if self.args.negate and self.args.closer:
                    print("Subtracting...")
                    self.curr_dist_mat -= w_j_nom*curr_Z_nom
                else:
                    self.curr_dist_mat += w_j_nom*curr_Z_nom
                if self.args.normalize:
                    print("Normalizing...")
                    self.curr_dist_mat = cov2corr(self.curr_dist_mat)
            else:
                if self.args.simple:
                    dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified(args=self.args, itr=j, closer=self.args.closer)
                    #dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified_anom(args=self.args, itr=j)
                elif self.args.simple_anom:
                    dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified_anom(args=self.args, itr=j)
                else:
                    dists, curr_Z, curr_triplet = self.replicate_user_feedback(args=self.args, itr=j)
                met_count = self.calc_sigma_metric(dists=dists)
                if not is_symmetric(curr_Z):
                    curr_Z = (curr_Z + curr_Z.T)/2
                self.triplets.append(curr_triplet)
                #self.u_r_arr.append(1.0/(len(self.u_r_arr)+1))
                if len(self.u_r_arr) >= 1:
                    self.u_r_arr.append(np.array(self.u_r_arr).sum()/len(self.u_r_arr))
                else:
                    self.u_r_arr.append(1.0)

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
                if self.args.negate:
                    w_j = self.bisection_search_posdef(curr_Z=curr_Z, curr_dist_mat=self.curr_dist_mat, w_l=0.0, w_u=w_j)

                #print("W_j {}".format(w_j))
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
                    #print("U{} {}".format(r, u_r_new))
                    new_u.append(u_r_new)
                new_u = np.array(new_u)
                new_u = new_u / new_u.sum()
                self.u_r = new_u
                self.u_r_arr = self.u_r.tolist()
            # print(self.u_r)
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
                self.curr_dist_mat += w_j*curr_Z


                if self.args.normalize:
                    print("Normalizing...")
                    self.curr_dist_mat = cov2corr(self.curr_dist_mat)
                # for i in range(len(w_s)):
                #     self.curr_dist_mat += np.array(w_s)[i]*np.array(Z_s)[i]

                # margin calculations
                #print("\nCurrent Margins...")
                for i in range(len(A_arr)):
                    H_i = np.trace(A_arr[i] @ self.curr_dist_mat.T)
                    #print("   >",H_i)

            sns.heatmap(self.curr_dist_mat)
            plt.savefig('figures/debugging_figs/X_{}_{}.png'.format(self.args.data, j))
            plt.close()
            #print()
            curr_f1, curr_precision, curr_recall, num_preds = final_classification(features=self.data, labels=self.labels, X=self.curr_dist_mat)
            self.f1s.append(curr_f1)
            self.precisions.append(curr_precision)
            self.recalls.append(curr_recall)
            self.num_preds.append(num_preds)
            print()
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

        original_dists = []
        orig_mu = np.mean(self.original_data, axis=0)
        for i in range(self.original_data.shape[0]):
            a_i = self.original_data[i, :]
            curr_dist = mahalanobis(a_i, orig_mu, self.curr_dist_mat)
            original_dists.append(curr_dist)
        original_dists = np.array(original_dists)
        original_anom_dists = original_dists[self.original_labels == 0]
        original_nominal_dists = original_dists[self.original_labels == 1]
        # save histogram of distances
        sns.histplot(original_anom_dists, alpha=0.5, label='anomalies', kde=True)
        sns.histplot(original_nominal_dists, alpha=0.5, label='nominal', kde=True)
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
        #print("Classification Training Score {}".format(clf.score(D_tot, label_tot)))

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
        #print("Dist a1->a2", mahalanobis(a_1, a_2, self.curr_dist_mat))
        #print("Dist a1->ak", mahalanobis(a_1, a_k, self.curr_dist_mat))
        #print("Dist a2->ak", mahalanobis(a_2, a_k, self.curr_dist_mat))

        self.data = np.delete(self.data, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0)
        self.labels = np.delete(self.labels, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0)
        
        return np.delete(dists, [selected_anom_idx, selected_a1_idx, selected_a2_idx], axis=0), curr_Z, curr_triplet
    
    def replicate_user_feedback_simplified(self, args, itr=0, use_mahal=False, closer=True):
        dists = []
        mu = np.mean(self.data, axis=0)
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, mu, self.curr_dist_mat)
            dists.append(curr_dist)
        dists = np.array(dists)
        anom_dists = dists[self.labels == 0]
        nominal_dists = dists[self.labels == 1]

        original_dists = []
        orig_mu = np.mean(self.original_data, axis=0)
        for i in range(self.original_data.shape[0]):
            a_i = self.original_data[i, :]
            curr_dist = mahalanobis(a_i, orig_mu, self.curr_dist_mat)
            original_dists.append(curr_dist)
        original_dists = np.array(original_dists)
        original_anom_dists = original_dists[self.original_labels == 0]
        original_nominal_dists = original_dists[self.original_labels == 1]
        # save histogram of distances
        sns.histplot(original_anom_dists, alpha=0.5, label='anomalies', kde=True)
        sns.histplot(original_nominal_dists, alpha=0.5, label='nominal', kde=True)
        plt.legend(loc='best')
        plt.xlabel('Distance to Mean')
        plt.tight_layout()
        plt.savefig('./figures/debugging_figs/dist_hist_simpl_{}_{}'.format(args.data, itr))
        plt.close()

        anom_feats = self.data[self.labels == 0, :]
        anom_dists_mean = np.mean(anom_dists)
        nom_feats = self.data[self.labels == 1, :]
        nom_dists_mean = np.mean(nominal_dists)
        # choose nominal point closest to the mean
        a_k_indx = np.argmin(np.abs(nominal_dists - nom_dists_mean))
        selected_nom_idx = np.where(self.labels==1)[0][a_k_indx]
        a_k = nom_feats[a_k_indx, :]
        
        # construct triplet
        # get anomaly 
        anom_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
        selected_anom_idx = np.where(self.labels==0)[0][anom_indx]
        #a_k_indx = np.argmin(anom_dists)
        anom_point = anom_feats[anom_indx, :]
        # calculate top_k most important features
        if closer:
            diff = a_k - mu
            if use_mahal:
                mahal_grad = self.curr_dist_mat @ diff
            else:
                mahal_grad = diff
            u_vec = mahal_grad/np.linalg.norm(mahal_grad)        
            coef_abs = np.abs(u_vec)
            coef_topk_idx = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]
            random_vector = np.random.rand(len(u_vec))
            z_vec = find_orthogonal_vector(random_vector, u_vec)
            #z_vec = u_vec
        else:
            print("PUSHING ANOMALY AWAY")
            diff = a_k - anom_point
            if use_mahal:
                mahal_grad = self.curr_dist_mat @ diff
            else:
                mahal_grad = diff
            u_vec = mahal_grad/np.linalg.norm(mahal_grad)
            z_vec = u_vec
        # construct weak learner from vector of most important features
        #z_vec = u_vec
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        #print(z_vec)
        curr_Z = z_vec@z_vec.T
        a_1 = mu
        a_2 = anom_point
        curr_triplet = (a_k, a_1, a_2) # a_k is the nominal point, a_1 is the mean, a_2 is the anomaly point
        #print("Dist a1->a2", mahalanobis(a_1, a_2, self.curr_dist_mat))
        #print("Dist a1->ak", mahalanobis(a_1, a_k, self.curr_dist_mat))
        #print("Dist a2->ak", mahalanobis(a_2, a_k, self.curr_dist_mat))

        self.data = np.delete(self.data, [selected_nom_idx, selected_anom_idx], axis=0)
        self.labels = np.delete(self.labels, [selected_nom_idx, selected_anom_idx], axis=0)
        
        return np.delete(dists, [selected_nom_idx, selected_anom_idx], axis=0), curr_Z, curr_triplet
    
    def replicate_user_feedback_double(self, args, itr=0, use_mahal=False, closer=True, negate=False):
        dists = []
        #mu = np.mean(self.data[np.where(self.labels==1)[0], :], axis=0)
        mu = np.mean(self.data, axis=0)
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, mu, self.curr_dist_mat)
            dists.append(curr_dist)
        dists = np.array(dists)
        anom_dists = dists[self.labels == 0]
        nominal_dists = dists[self.labels == 1]

        original_dists = []
        orig_mu = np.mean(self.original_data, axis=0)
        for i in range(self.original_data.shape[0]):
            a_i = self.original_data[i, :]
            curr_dist = mahalanobis(a_i, orig_mu, self.curr_dist_mat)
            original_dists.append(curr_dist)
        original_dists = np.array(original_dists)
        original_anom_dists = original_dists[self.original_labels == 0]
        original_nominal_dists = original_dists[self.original_labels == 1]
        # save histogram of distances
        sns.histplot(original_anom_dists, alpha=0.5, label='anomalies', kde=True)
        sns.histplot(original_nominal_dists, alpha=0.5, label='nominal', kde=True)
        plt.legend(loc='best')
        plt.xlabel('Distance to Mean')
        plt.tight_layout()
        plt.savefig('./figures/debugging_figs/dist_hist_double_{}_{}'.format(args.data, itr))
        plt.close()

        anom_feats = self.data[self.labels == 0, :]
        anom_dists_mean = np.mean(anom_dists)
        nom_feats = self.data[self.labels == 1, :]
        nom_dists_mean = np.mean(nominal_dists)
        # choose nominal point closest to the mean of nominal distances
        a_k_indx = np.argmin(np.abs(nominal_dists - nom_dists_mean))
        selected_nom_idx = np.where(self.labels==1)[0][a_k_indx]
        a_k = nom_feats[a_k_indx, :]
        

        # construct triplet
        # get anomaly closest to the mean of anomaly distances
        anom_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
        selected_anom_idx = np.where(self.labels==0)[0][anom_indx]
        #a_k_indx = np.argmin(anom_dists)
        anom_point = anom_feats[anom_indx, :]

        # anomaly section
        diff = anom_point - mu
        if use_mahal:
            mahal_grad = self.curr_dist_mat @ diff
        else:
            mahal_grad = diff
        u_vec = mahal_grad/np.linalg.norm(mahal_grad)
        coef_abs = np.abs(u_vec)
        coef_topk_idx = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]

        # construct weak learner from vector of most important features
        # z_vec = np.zeros_like(u_vec)
        # z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
        z_vec = u_vec
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        curr_Z_anom = z_vec@z_vec.T

        # nominal section
        if closer:
            diff = a_k - mu
            if use_mahal:
                mahal_grad = self.curr_dist_mat @ diff
            else:
                mahal_grad = diff
            u_vec = mahal_grad/np.linalg.norm(mahal_grad)      
            coef_abs = np.abs(u_vec)
            coef_topk_idx = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]
            random_vector = np.random.rand(len(u_vec))
            z_vec = find_orthogonal_vector(random_vector, u_vec)
            if negate:
                z_vec = u_vec
        else:
            print("PUSHING ANOMALY AWAY")
            diff = a_k - anom_point
            if use_mahal:
                mahal_grad = self.curr_dist_mat @ diff
            else:
                mahal_grad = diff
            u_vec = mahal_grad/np.linalg.norm(mahal_grad)
            z_vec = u_vec

        # construct weak learner from vector of most important features
        # z_vec = np.zeros_like(u_vec)
        # z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        #print(z_vec)
        curr_Z_nom = z_vec@z_vec.T

        a_1 = mu
        a_2 = anom_point
        # TODO - get another nominal point instead of using mean
        # TODO - get another anomaly point instead of using mean? - set up positive pair of anomalies?

        curr_triplet = (a_k, a_1, a_2) # a_k is the nominal point, a_1 is the mean, a_2 is the anomaly point
        second_triplet = (a_1, a_k, a_2) # a_1 is the mean, a_k is the nominal point, a_2 is the anomaly point
        #print("Dist a1->a2", mahalanobis(a_1, a_2, self.curr_dist_mat))
        #print("Dist a1->ak", mahalanobis(a_1, a_k, self.curr_dist_mat))
        #print("Dist a2->ak", mahalanobis(a_2, a_k, self.curr_dist_mat))

        self.data = np.delete(self.data, [selected_nom_idx, selected_anom_idx], axis=0)
        self.labels = np.delete(self.labels, [selected_nom_idx, selected_anom_idx], axis=0)
        
        return np.delete(dists, [selected_nom_idx, selected_anom_idx], axis=0), curr_Z_anom, curr_Z_nom, curr_triplet, second_triplet
    
    def replicate_user_feedback_simplified_anom(self, args, itr=0, use_mahal=False):
        dists = []
        mu = np.mean(self.data, axis=0)
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, mu, self.curr_dist_mat)
            dists.append(curr_dist)
        dists = np.array(dists)
        anom_dists = dists[self.labels == 0]
        nominal_dists = dists[self.labels == 1]

        original_dists = []
        orig_mu = np.mean(self.original_data, axis=0)
        for i in range(self.original_data.shape[0]):
            a_i = self.original_data[i, :]
            curr_dist = mahalanobis(a_i, orig_mu, self.curr_dist_mat)
            original_dists.append(curr_dist)
        original_dists = np.array(original_dists)
        original_anom_dists = original_dists[self.original_labels == 0]
        original_nominal_dists = original_dists[self.original_labels == 1]
        # save histogram of distances
        sns.histplot(original_anom_dists, alpha=0.5, label='anomalies', kde=True)
        sns.histplot(original_nominal_dists, alpha=0.5, label='nominal', kde=True)
        plt.legend(loc='best')
        plt.xlabel('Distance to Mean')
        plt.tight_layout()
        plt.savefig('./figures/debugging_figs/dist_hist_anom_{}_{}'.format(args.data, itr))
        plt.close()

        anom_feats = self.data[self.labels == 0, :]
        anom_dists_mean = np.mean(anom_dists)
        nom_feats = self.data[self.labels == 1, :]
        nom_dists_mean = np.mean(nominal_dists)
        # choose nominal point closest to the mean of nominal distances
        a_k_indx = np.argmin(np.abs(nominal_dists - nom_dists_mean))
        selected_nom_idx = np.where(self.labels==1)[0][a_k_indx]
        a_k = nom_feats[a_k_indx, :]
        

        # construct triplet
        # get anomaly closest to the mean of anomaly distances
        anom_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
        selected_anom_idx = np.where(self.labels==0)[0][anom_indx]
        #a_k_indx = np.argmin(anom_dists)
        anom_point = anom_feats[anom_indx, :]

        # calculate top_k most important features
        diff = anom_point - mu
        if use_mahal:
            mahal_grad = self.curr_dist_mat @ diff
        else:
            mahal_grad = diff
        u_vec = mahal_grad/np.linalg.norm(mahal_grad)
        coef_abs = np.abs(u_vec)
        coef_topk_idx = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]

        # construct weak learner from vector of most important features
        # z_vec = np.zeros_like(u_vec)
        # z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
        z_vec = u_vec
        z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
        z_vec = np.expand_dims(z_vec, 1)
        #print(z_vec)
        curr_Z = z_vec@z_vec.T
        a_1 = mu
        a_2 = anom_point

        # TODO - get another nominal point instead of using mean
        # TODO - get another anomaly point instead of using mean? - set up positive pair of anomalies?

        curr_triplet = (a_k, a_1, a_2) # a_k is the nominal point, a_1 is the mean, a_2 is the anomaly point
        second_triplet = (a_1, a_k, a_2) # a_1 is the mean, a_k is the nominal point, a_2 is the anomaly point
        #print("Dist a1->a2", mahalanobis(a_1, a_2, self.curr_dist_mat))
        #print("Dist a1->ak", mahalanobis(a_1, a_k, self.curr_dist_mat))
        #print("Dist a2->ak", mahalanobis(a_2, a_k, self.curr_dist_mat))

        self.data = np.delete(self.data, [selected_nom_idx, selected_anom_idx], axis=0)
        self.labels = np.delete(self.labels, [selected_nom_idx, selected_anom_idx], axis=0)
        
        return np.delete(dists, [selected_nom_idx, selected_anom_idx], axis=0), curr_Z, curr_triplet


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
    
    def bisection_search_posdef(self, curr_Z, curr_dist_mat, w_l=1e-5, w_u=1.0, eps=1e-5):
        w_j = 0.0
        while w_u - w_l >= eps:
            w_j = 0.5*(w_u + w_l)
            proposed_dist_mat = curr_dist_mat - w_j*curr_Z
            if is_pos_def(proposed_dist_mat):
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
    
    def get_us(self):
        return self.u_r_arr
    
    def calc_sigma_metric(self, dists):
        three_sigma_empirical = 3*dists.var()
        # num_points_outside = (dists > three_sigma_empirical).sum()
        top_40_far_idx = np.argpartition(dists, -40)[-40:]
        top_40_far_labels = self.labels[top_40_far_idx]
        top_40_close_idx = np.argpartition(dists, 40)[:40]
        top_40_close_labels = self.labels[top_40_close_idx]
        #print("Farthest 40 Points Anomalies {}".format((top_40_far_labels == 0).sum()))
        #print("Closest 40 Points Anomalies {}".format((top_40_close_labels == 0).sum()))
        #print("Three Sigma {}".format(three_sigma_empirical))
        #print("Dist Mean {} Min {} Max {}".format(dists.mean(), dists.min(), dists.max()))

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

def init_covar(data, normalize=False):
    A_t = np.cov(data.T)
    #A_t = np.corrcoef(data.T)
    A_t = A_t + 1e-5*np.eye(A_t.shape[0])

    if normalize:
        A_t = cov2corr(A_t)
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

def find_orthogonal_vector(v, reference):
    """Find an orthogonal vector to the given vector v using the reference vector."""
    v_orthogonal = v - (np.dot(v, reference) / np.dot(reference, reference)) * reference
    return v_orthogonal

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

def final_classification(features, labels, X):
    dists = []
    chisq_vals = []
    mu = np.mean(features, axis=0)
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = mahalanobis(a_i, mu, X)
        dists.append(curr_dist)
        chisq_vals.append(chi2.sf(curr_dist, df=features.shape[1]-1))
    dists = np.array(dists)
    chisq_vals = np.array(chisq_vals)
    sigma = dists.std()
    preds_chisq = np.zeros_like(labels)
    #print(chisq_vals)
    preds_chisq[chisq_vals <= 0.1] = 1
    #print("ChiSq: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(labels, preds_chisq), f1_score(labels, preds_chisq), precision_score(labels, preds_chisq), recall_score(labels, preds_chisq)))
    #print("Avg Precision Score {}".format(average_precision_score(labels, chisq_vals)))

    # class zero is anomaly, switch it for sake of calculation
    cf1 = f1_score(1-labels, preds_chisq)
    cprec = precision_score(1-labels, preds_chisq)
    crec = recall_score(1-labels, preds_chisq)
    print("Number Predicted Anomalies {}".format(preds_chisq.sum()))

    return cf1, cprec, crec, preds_chisq.sum()

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

def fast_kernel(X1, X2):
    return pairwise_distances(X1, X2, metric=lambda x, y: np.dot(x, y))


if __name__ == "__main__":
    args = parse_arguments()
    #main(args)
    #main_boost(args)

    data, features, labels = resolve_data(args)
    # unique count of labels
    print("Unique Labels {}".format(np.unique(labels, return_counts=True)))
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
        if args.data == 'bank':
            labels = 1-labels # switch 0 and 1
    else:
        labels[labels != 0] = 1

    features = scaler.fit_transform(features)
    # randomly select 10000 samples from features
    labels_svm = labels.copy()
    labels_svm[labels_svm == 0] = -1
    features_subset, labels_subset = shuffle(features, labels_svm, random_state=42)
    features_subset = features_subset[:10000]
    labels_subset = labels_subset[:10000]
    ####################
    # svm_boost = OCSVMBoost(data=features_subset, m=30)
    # hypotheses, lg_mults = svm_boost.fit()
    # svm_boost_preds = []
    # print("Number of Hypotheses {}".format(len(hypotheses)))
    # for i, h in enumerate(hypotheses):
    #     print(">> Enumerating Hypothesis {} <<".format(i))
    #     cpreds = h.predict(features)
    #     svm_boost_preds.append(cpreds*lg_mults[i])
    # print()
    # svm_boost_preds = np.array(svm_boost_preds)
    # svm_boost_preds = np.sign(np.sum(svm_boost_preds, axis=0)).T
    # svm_boost_preds[svm_boost_preds == 1] = 0 # nominals
    # svm_boost_preds[svm_boost_preds == -1] = 1 # outliers
    # print(svm_boost_preds.shape)


    #####################
    # print("Fitting OC-SVM {}".format(features_subset.shape))
    # clf = OneClassSVM(kernel='linear', nu=0.1, shrinking=False)
    # clf.fit(features_subset)
    # print("Predicting OC-SVM")
    # svm_preds = clf.predict(features)
    # svm_preds[svm_preds == 1] = 0  # nominals
    # svm_preds[svm_preds == -1] = 1 # outliers
    # #print(-clf.intercept_[0] / (clf.nu * len(features)))
    # #print(clf.dual_coef_.ravel() / (clf.nu * len(features)))

    # clf = OCSVMCVXPrimal(data=features_subset, v=0.2)
    # w_primal, rho_primal, xi_primal = clf.solve()
    # #print(rho_primal)
    # #print("W Primal {}".format(w_primal))
    # #print("Rho Primal {}".format(rho_primal))
    # #print("Xi Primal {}".format(xi_primal))
    # svm_preds_primal = clf.predict(features)
    # svm_preds_primal[svm_preds_primal == 1] = 0  # nominals
    # svm_preds_primal[svm_preds_primal == -1] = 1 # outliers
    # # repeat for the dual
    # clf = OCSVMCVXDual(data=features_subset, v=0.2)
    # alpha_dual = clf.solve()
    # #print("Alpha Dual {}".format(alpha_dual))
    # svm_preds_dual = clf.predict(features)
    # svm_preds_dual[svm_preds_dual == 1] = 0  # nominals
    # svm_preds_dual[svm_preds_dual == -1] = 1 # outliers
    # #print(clf.alpha.value)

    # clf_rad = OCSVMCVXPrimalRad(data=features_subset, v=0.1)
    # center_rad, radius_rad, xi_rad = clf_rad.solve()
    # #print(center_rad, radius_rad)
    # svm_preds_rad = clf_rad.predict(features)
    # svm_preds_rad[svm_preds_rad == 1] = 0  # nominals
    # svm_preds_rad[svm_preds_rad == -1] = 1 # outliers

    # clf = OCSVMCVXDualRad(data=features_subset, v=0.2)
    # alpha_rad = clf.solve()
    # svm_preds_rad_dual = clf.predict(features)
    # svm_preds_rad_dual[svm_preds_rad_dual == 1] = 0  # nominals
    # svm_preds_rad_dual[svm_preds_rad_dual == -1] = 1 # outliers

    # clf_rad_alt = OCSVMRadAlt(v=0.1, n_features=features_subset.shape[1])
    # ret_dict_rad = clf_rad_alt.fit(features_subset)
    # svm_preds_rad_alt = clf_rad_alt.predict(features)
    # svm_preds_rad_alt[svm_preds_rad_alt == 1] = 0  # nominals
    # svm_preds_rad_alt[svm_preds_rad_alt == -1] = 1 # outliers

    # clf_primal_min = OCSVMCVXPrimalMinimization(nu=0.3)
    # clf_primal_min.fit(features_subset)
    # svm_preds_primal_min = clf_primal_min.predict(features)
    # svm_preds_primal_min[svm_preds_primal_min == 1] = 0  # nominals
    # svm_preds_primal_min[svm_preds_primal_min == -1] = 1 # outliers

    # clf_primal_mix = OCSVMMix(v=0.05, kernel_approx=True, gamma=0.5, n_features=200)
    # clf_primal_mix.fit(features_subset, labels_subset)
    # svm_preds_primal_mix = clf_primal_mix.predict(features_subset)
    # svm_preds_primal_mix[svm_preds_primal_mix == 1] = 0  # nominals
    # svm_preds_primal_mix[svm_preds_primal_mix == -1] = 1 # outliers



    # mu_support, idx_support = ocsvm_solver(K=np.dot(features_subset, features_subset.T), nu=0.1)
    # calc_rho = compute_rho(K=np.dot(features_subset, features_subset.T), mu_support=mu_support, idx_support=idx_support)
    # X_support = features_subset[idx_support]
    # G = np.dot(features, X_support.T)
    # decision = G.dot(mu_support) - calc_rho
    # svm_preds_cvxopt = np.sign(decision)
    # svm_preds_cvxopt[svm_preds_cvxopt == 1] = 0  # nominals
    # svm_preds_cvxopt[svm_preds_cvxopt == -1] = 1 # outliers

    ##############################
    # iterate over a few points and give the true labels to the dual
    # we do this by selecting the top 10 closest points to the decision boundary that are incorrectly classified
    # and then we give the true labels to the dual
    
    # select the top 10 closest points to the decision boundary that are incorrectly classified

    ####################
    # labels_subset[labels_subset == -1] = 0
    # incorrect_points = (svm_preds_primal_mix != 1-labels_subset)
    # incorrect_points_idx = np.where(incorrect_points)[0]
    # decision_func_mix = clf_primal_mix.decision_function(features_subset)
    # smallest_k_idx = np.argpartition(np.abs(decision_func_mix[incorrect_points_idx]), 55)[:55]
    # original_idx = incorrect_points_idx[smallest_k_idx]
    # mix_precs = []
    # print("Mix Ratio: ", svm_preds_primal_mix.sum()/svm_preds_primal_mix.shape[0])
    # #print("SVM Mix: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_primal_mix), f1_score(1-labels, svm_preds_primal_mix), precision_score(1-labels, svm_preds_primal_mix), recall_score(1-labels, svm_preds_primal_mix)))
    # print("SVM Mix: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels_subset, svm_preds_primal_mix), f1_score(1-labels_subset, svm_preds_primal_mix), precision_score(1-labels_subset, svm_preds_primal_mix), recall_score(1-labels_subset, svm_preds_primal_mix)))
    # for j, idx in enumerate(original_idx):
    #     print(">> Iteration {} <<".format(j))
    #     print("True Label {}".format(1-labels_subset[idx]))
    #     print("Index {}".format(idx))
    #     clf_primal_mix.add_labeled_example(idx)
    #     svm_preds_primal_mix = clf_primal_mix.predict(features)
    #     svm_preds_primal_mix[svm_preds_primal_mix == 1] = 0  # nominals
    #     svm_preds_primal_mix[svm_preds_primal_mix == -1] = 1 # outliers
    #     # print the ratio of anomalies
    #     print("Mix Ratio: ", svm_preds_primal_mix.sum()/svm_preds_primal_mix.shape[0])
    #     # print the accuracy, f1, precision, recall
    #     mix_precs.append(precision_score(1-labels, svm_preds_primal_mix))
    #     print("SVM Mix: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_primal_mix), f1_score(1-labels, svm_preds_primal_mix), precision_score(1-labels, svm_preds_primal_mix), recall_score(1-labels, svm_preds_primal_mix)))
    #     print(">>>>>>>>><<<<<<<<<")

    #     # check incorrect points again and see if classified correctly
    #     print((svm_preds_primal_mix[idx] == (1-labels)[idx]))
    # plt.plot(mix_precs)
    # plt.xlabel('Iteration')
    # plt.ylabel('Precision')
    # plt.title('SemiSupervised SVM Precision {}'.format(args.data))
    # plt.ylim(0.0, 1.0)
    # plt.savefig('./figures/debugging_figs/semi_supervised_svm_precision_{}'.format(args.data))
    ################

    ####################
    # incorrect_points = (svm_preds_rad != 1-labels)
    # incorrect_points_idx = np.where(incorrect_points)[0]
    # decision_func_rad = clf_rad.decision_function(features)
    # smallest_k_idx = np.argpartition(np.abs(decision_func_rad[incorrect_points_idx]), 10)[:10]
    # original_idx = incorrect_points_idx[smallest_k_idx]
    # print("Radial Ratio: ", svm_preds_rad.sum()/svm_preds_rad.shape[0])
    # print("SVM Radial: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad), f1_score(1-labels, svm_preds_rad), precision_score(1-labels, svm_preds_rad), recall_score(1-labels, svm_preds_rad)))
    # for j, idx in enumerate(original_idx):
    #     print(">> Iteration {} <<".format(j))
    #     curr_label = 1-labels[idx]
    #     if curr_label == 0: # nominal
    #         center_rad, radius_rad, xi_rad = clf_rad.add_hard_constraint(idx, 1)
    #     else:
    #         center_rad, radius_rad, xi_rad = clf_rad.add_hard_constraint(idx, -1)
    #     svm_preds_rad = clf_rad.predict(features)
    #     svm_preds_rad[svm_preds_rad == 1] = 0  # nominals
    #     svm_preds_rad[svm_preds_rad == -1] = 1 # outliers
    #     # print the ratio of anomalies
    #     print("Radial Ratio: ", svm_preds_rad.sum()/svm_preds_rad.shape[0])
    #     # print the accuracy, f1, precision, recall
    #     print("SVM Radial: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad), f1_score(1-labels, svm_preds_rad), precision_score(1-labels, svm_preds_rad), recall_score(1-labels, svm_preds_rad)))
    #     print(">>>>>>>>><<<<<<<<<")

    #     # check incorrect points again and see if classified correctly
    #     print((svm_preds_rad[idx] == (1-labels)[idx]))

    ################


    # incorrect_points = (svm_preds_rad_alt != 1-labels)
    # incorrect_points_idx = np.where(incorrect_points)[0]
    # decision_func_rad_alt = clf.decision_function(features)
    # smallest_k_idx = np.argpartition(np.abs(decision_func_rad_alt[incorrect_points_idx]), 10)[:10]
    # original_idx = incorrect_points_idx[smallest_k_idx]
    # print("Radial Alt Ratio: ", svm_preds_rad_alt.sum()/svm_preds_rad_alt.shape[0])
    # print("SVM Radial Alt: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad_alt), f1_score(1-labels, svm_preds_rad_alt), precision_score(1-labels, svm_preds_rad_alt), recall_score(1-labels, svm_preds_rad_alt)))
    # for j, idx in enumerate(original_idx):
    #     print(">> Iteration {} <<".format(j))
    #     curr_label = 1-labels[idx]
    #     if curr_label == 0: # nominal
    #         clf_rad_alt.update_constraints(idx, 1)
    #     else: # anomaly
    #         clf_rad_alt.update_constraints(idx, -1)
    #     svm_preds_rad_alt = clf_rad_alt.predict(features)
    #     svm_preds_rad_alt[svm_preds_rad_alt == 1] = 0  # nominals
    #     svm_preds_rad_alt[svm_preds_rad_alt == -1] = 1 # outliers
    #     # print the ratio of anomalies
    #     print("Radial Alt Ratio: ", svm_preds_rad_alt.sum()/svm_preds_rad_alt.shape[0])
    #     # print the accuracy, f1, precision, recall
    #     print("SVM Radial Alt: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad_alt), f1_score(1-labels, svm_preds_rad_alt), precision_score(1-labels, svm_preds_rad_alt), recall_score(1-labels, svm_preds_rad_alt)))
    #     print(">>>>>>>>><<<<<<<<<")

    #     # check incorrect points again and see if classified correctly
    #     print((svm_preds_rad_alt[idx] == (1-labels)[idx]))
    ######################

        



    # print(svm_preds_primal)
    # print(svm_preds_dual)
    ########
    # print("SKLearn Ratio: ", svm_preds.sum()/svm_preds.shape[0])
    # print("Primal Ratio: ", svm_preds_primal.sum()/svm_preds_primal.shape[0])
    # print("Dual Ratio: ", svm_preds_dual.sum()/svm_preds_dual.shape[0])
    # print("Radial Ratio: ", svm_preds_rad.sum()/svm_preds_rad.shape[0])
    # print("Radial Dual Ratio: ", svm_preds_rad_dual.sum()/svm_preds_rad_dual.shape[0])
    # print("CVXOPT Ratio: ", svm_preds_cvxopt.sum()/svm_preds_cvxopt.shape[0])
    # print("Radial Alt Ratio: ", svm_preds_rad_alt.sum()/svm_preds_rad_alt.shape[0])
    # print("Primal Min Ratio: ", svm_preds_primal_min.sum()/svm_preds_primal_min.shape[0])
    # print("Primal Mix Ratio: ", svm_preds_primal_mix.sum()/svm_preds_primal_mix.shape[0])




    #print(svm_preds_dual)
    # get the distance from the decision boundary
    # get the true labels


    ############
    # print("Fitting Naive OC-SVM {}".format(features_subset.shape))
    # clf_naive = NaiveBoostedOneClassSVM(n_estimators=len(hypotheses))
    # clf_naive.fit(features_subset)
    # naive_preds = clf_naive.predict(features)
    # naive_preds[naive_preds == 1] = 0  # nominals
    # naive_preds[naive_preds == -1] = 1 # outliers

    init_dist_mat = init_covar(features, normalize=args.normalize)
    print("Data {} Shape {}".format(args.data, features.shape))

    bm = BoostMetric(data=features, labels=labels, init_dist_mat=init_dist_mat, args=args, v=args.v, J=args.iters, top_k=args.k)
    X = bm.iterate()
    w, Z = bm.get_w_Z()
    us = bm.get_us()
    argmax_u = np.argmax(us, axis=0)
    print("Argmax U {}".format(us[argmax_u]))
    print("Argmax of U in W {}".format(w[argmax_u]))
    #print(us)
    exit()
    print("Length of w {}".format(len(w)))
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
    ###############
    
    # print("\nSVM: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds), f1_score(1-labels, svm_preds), precision_score(1-labels, svm_preds), recall_score(1-labels, svm_preds)))
    # print("SVM Primal: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_primal), f1_score(1-labels, svm_preds_primal), precision_score(1-labels, svm_preds_primal), recall_score(1-labels, svm_preds_primal)))
    # print("SVM Dual: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_dual), f1_score(1-labels, svm_preds_dual), precision_score(1-labels, svm_preds_dual), recall_score(1-labels, svm_preds_dual)))
    # print("SVM Radial: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad), f1_score(1-labels, svm_preds_rad), precision_score(1-labels, svm_preds_rad), recall_score(1-labels, svm_preds_rad)))
    # print("SVM Radial Dual: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad_dual), f1_score(1-labels, svm_preds_rad_dual), precision_score(1-labels, svm_preds_rad_dual), recall_score(1-labels, svm_preds_rad_dual)))
    # print("SVM CVXOPT: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_cvxopt), f1_score(1-labels, svm_preds_cvxopt), precision_score(1-labels, svm_preds_cvxopt), recall_score(1-labels, svm_preds_cvxopt)))
    # print("SVM Radial Alt: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_rad_alt), f1_score(1-labels, svm_preds_rad_alt), precision_score(1-labels, svm_preds_rad_alt), recall_score(1-labels, svm_preds_rad_alt)))
    # print("SVM Primal Min: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_primal_min), f1_score(1-labels, svm_preds_primal_min), precision_score(1-labels, svm_preds_primal_min), recall_score(1-labels, svm_preds_primal_min)))
    # print("SVM Primal Mix: Accuracy {}, F1 {}, Precision {}, Recall {}".format(accuracy_score(1-labels, svm_preds_primal_mix), f1_score(1-labels, svm_preds_primal_mix), precision_score(1-labels, svm_preds_primal_mix), recall_score(1-labels, svm_preds_primal_mix)))
    #########
    #print("SVM Boost: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, svm_boost_preds), precision_score(1-labels, svm_boost_preds), recall_score(1-labels, svm_boost_preds)))
    #print("SVM Naive Boost: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, naive_preds), precision_score(1-labels, naive_preds), recall_score(1-labels, naive_preds)))
    print("BoostMetric: F1 {}, Precision {}, Recall {}".format(bm.f1s[-1], bm.precisions[-1], bm.recalls[-1]))
    #total_distances = mahal_all_points(features, X)
    #knn_clf = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    #knn_clf.fit(total_distances, labels)
    #knn_preds = knn_clf.predict(total_distances)
    #print("KNN: F1 {}, Precision {}, Recall {}".format(f1_score(labels, knn_preds), precision_score(labels, knn_preds), recall_score(labels, knn_preds)))
    #knn_clf_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    #knn_clf_euclidean.fit(features, labels)
    #knn_preds_euclidean = knn_clf_euclidean.predict(features)
    #print("KNN Euclidean: F1 {}, Precision {}, Recall {}".format(f1_score(labels, knn_preds_euclidean), precision_score(labels, knn_preds_euclidean), recall_score(labels, knn_preds_euclidean)))
    preds_mahal_percentile = predict_percentile(args, features, X, labels=labels, percentile=95, dist='mahalanobis')
    print("Mahal Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_mahal_percentile), precision_score(1-labels, 1-preds_mahal_percentile), recall_score(1-labels, 1-preds_mahal_percentile)))
    preds_euclidean_percentile = predict_percentile(args, features, X=None, labels=labels, percentile=95, dist='euclidean')
    print("Euclidean Percentile: F1 {}, Precision {}, Recall {}".format(f1_score(1-labels, 1-preds_euclidean_percentile), precision_score(1-labels, 1-preds_euclidean_percentile), recall_score(1-labels, 1-preds_euclidean_percentile)))

    dists = []
    mu = np.mean(features, axis=0)
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = np.linalg.norm(a_i - mu)
        dists.append(curr_dist)
    dists = np.array(dists)
    anom_dists = dists[labels == 0]
    nominal_dists = dists[labels == 1]

    # save histogram of distances
    sns.histplot(anom_dists, alpha=0.5, label='anomalies', kde=True)
    sns.histplot(nominal_dists, alpha=0.5, label='nominal', kde=True)
    plt.legend(loc='best')
    plt.xlabel('Distance to Mean')
    plt.tight_layout()
    plt.savefig('./figures/debugging_figs/dist_hist_euclid_{}'.format(args.data))
    plt.close()

    ##########

    #final_classification(features=features, labels=labels, X=X)
    ##############################