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
#from autoencoder import AE, train_model
#from torch.utils.data import TensorDataset, DataLoader
#import torch

from oml_two import calc_ybar, calc_A_tpo

class BoostMetric:
    def __init__(self, data, labels, v, J, init_dist_mat, args, keep_top_k=False, top_k=3, seed=42):
        np.random.seed(seed)

        self.seed = seed
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

        self.queried_nominals = []
        self.queried_anomalies = []
        self.nominal_directions = []
        self.queried_indices = []
        self.masked_data = np.ones(self.data.shape[0], dtype=bool)

        self.num_anomalies_queried = 0
        self.num_anomalies_queried_list = []

        self.oml_flag = False # this is disgusting but it's a quick fix for now
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
            if self.args.true_feedback:
                self.oml_flag = False

                _ , curr_Z, curr_triplet = self.get_user_feedback(args=self.args, itr=j, use_orig_mu=self.args.original_mu, use_top_k=self.args.use_top_k, negate=self.args.negate, use_mahal=self.args.use_mahal)
                if not self.oml_flag:
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
                    w_j = self.bisection_search(curr_Z, H_s, w_l=1e-5, w_u=self.args.w_upper)
                    if self.args.negate:
                        w_j = self.bisection_search_posdef(curr_Z=curr_Z, curr_dist_mat=self.curr_dist_mat, w_l=0.0, w_u=w_j)
                    new_u = []

                    ################
                    # REMOVE ME
                    if self.args.opo:
                        w_j = 1.0
                    print("W_j {}".format(w_j))
                    ################

                    for r in range(self.u_r.shape[0]):
                        in_exp = -H_s[r]*w_j
                        in_exp = np.clip(in_exp, -100, 100)
                        u_r_new = self.u_r[r]*np.exp(in_exp) #overflow could happen here too potentially
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


                    #w_s.append(w_j)
                    #w_s.append(1.0)
                    #Z_s.append(curr_Z)
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
                self.num_anomalies_queried_list.append(self.num_anomalies_queried)


            elif self.args.double:
                dists, curr_Z_anom, curr_Z_nom, curr_triplet, second_triplet = self.replicate_user_feedback_double(args=self.args, itr=j, closer=self.args.closer, negate=self.args.negate, use_orig_mu=self.args.original_mu, use_top_k=self.args.use_top_k)
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
                    dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified(args=self.args, itr=j, closer=self.args.closer, use_orig_mu=self.args.original_mu)
                    #dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified_anom(args=self.args, itr=j)
                elif self.args.simple_anom:
                    dists, curr_Z, curr_triplet = self.replicate_user_feedback_simplified_anom(args=self.args, itr=j, use_orig_mu=self.args.original_mu, use_top_k=self.args.use_top_k)
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
            # curr_f1, curr_precision, curr_recall, num_preds = final_classification(features=self.data, labels=self.labels, X=self.curr_dist_mat)
            # self.f1s.append(curr_f1)
            # self.precisions.append(curr_precision)
            # self.recalls.append(curr_recall)
            # self.num_preds.append(num_preds)
            print()
        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        
        if self.args.true_feedback:
            # save number of anomalies queried list
            if self.args.save_suffix:
                np.savetxt('figures/debugging_figs_true/num_anomalies_queried_{}_{}.csv'.format(self.args.data, self.args.save_suffix), np.array(self.num_anomalies_queried_list).astype(int))
            else:
                np.savetxt('figures/debugging_figs_true/num_anomalies_queried_{}_{}.csv'.format(self.args.data, self.seed), np.array(self.num_anomalies_queried_list).astype(int))
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
    
    def replicate_user_feedback_simplified(self, args, itr=0, use_mahal=False, closer=True, use_orig_mu=True):
        dists = []
        if use_orig_mu:
            mu = np.mean(self.original_data, axis=0)
        else:
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
    
    def replicate_user_feedback_double(self, args, itr=0, use_mahal=False, closer=True, negate=False, use_orig_mu=True, use_top_k=True):
        dists = []
        #mu = np.mean(self.data[np.where(self.labels==1)[0], :], axis=0)
        if use_orig_mu:
            mu = np.mean(self.original_data, axis=0)
        else:
            mu = np.mean(self.data, axis=0)
        for i in range(self.data.shape[0]):
            a_i = self.data[i, :]
            curr_dist = mahalanobis(a_i, mu, self.curr_dist_mat)
            dists.append(curr_dist)
        dists = np.array(dists)
        anom_dists = dists[self.labels == 0]
        nominal_dists = dists[self.labels == 1]
        percentile_thresh = np.percentile(dists, 95)


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

        original_indexes = np.arange(dists.shape[0])
        anom_feats = self.data[self.labels == 0, :]
        anom_indexes = original_indexes[self.labels == 0]
        anom_dists_mean = np.mean(anom_dists)
        nom_feats = self.data[self.labels == 1, :]
        nom_indexes = original_indexes[self.labels == 1]
        nom_dists_mean = np.mean(nominal_dists)

        # choose anomaly point closest to the threshold, wrong side of boundary (negative)
        anom_dist_thresh = anom_dists - percentile_thresh
        negative_anom_dists = anom_dist_thresh[anom_dist_thresh < 0]
        if len(negative_anom_dists) > 0:
            negative_anom_indexes = anom_indexes[anom_dist_thresh < 0]
            closest_anom_indx = np.argmax(negative_anom_dists)
            selected_anom_idx = negative_anom_indexes[closest_anom_indx]
            anom_point = anom_feats[closest_anom_indx, :]
        else:
            # choose anomaly point closest to the mean of anomaly distances
            print("All Anomalies are on the correct side of the boundary")
            anom_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
            selected_anom_idx = np.where(self.labels==0)[0][anom_indx]
            anom_point = anom_feats[anom_indx, :]

        # choose nominal point closest to the threshold, wrong side of boundary (positive)
        nom_dist_thresh = nominal_dists - percentile_thresh
        positive_nom_dists = nom_dist_thresh[nom_dist_thresh > 0]
        if len(positive_nom_dists) > 0:
            positive_nom_indexes = nom_indexes[nom_dist_thresh > 0]
            closest_nom_indx = np.argmin(positive_nom_dists)
            selected_nom_idx = positive_nom_indexes[closest_nom_indx]
            a_k = nom_feats[closest_nom_indx, :]
        else:
            print("All Nominals are on the correct side of the boundary")
            # FIRST ONE HERE IS TO MEAN OF NOMINALS
            a_k_indx = np.argmin(np.abs(nominal_dists - nom_dists_mean))
            # SECOND ONE HERE IS FARTHEST FROM BOUNDARY
            #a_k_indx = np.argmax(np.abs(nom_dist_thresh))
            selected_nom_idx = np.where(self.labels==1)[0][a_k_indx]
            a_k = nom_feats[a_k_indx, :]

        # #################################
        # # TODO REMOVE ME I'M JUST A TESTER
        # selected_nom_idx = np.random.choice(np.where(self.labels==1)[0], size=1)[0]
        # a_k = self.data[selected_nom_idx, :]
        # # TODO REMOVE ME I'M JUST A TESTER
        # #################################


        self.queried_anomalies.append(anom_point)
        self.queried_nominals.append(a_k)

        ############################################################################################################
        # # ORIGINAL CODE
        # # construct triplet
        # # get anomaly closest to the mean of anomaly distances
        # anom_indx = np.argmin(np.abs(anom_dists - anom_dists_mean))
        # selected_anom_idx = np.where(self.labels==0)[0][anom_indx]
        # #a_k_indx = np.argmin(anom_dists)
        # anom_point = anom_feats[anom_indx, :]
        

        # # choose nominal point closest to the mean of nominal distances
        # a_k_indx = np.argmin(np.abs(nominal_dists - nom_dists_mean))
        # selected_nom_idx = np.where(self.labels==1)[0][a_k_indx]
        # a_k = nom_feats[a_k_indx, :]
        
        ############################################################################################################
        

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
        if use_top_k:
            print("Using Top K Anom")
            z_vec = np.zeros_like(u_vec)
            z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
        else:
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
            if use_top_k:
                print("Using Top K Nom")
                coef_abs = np.abs(u_vec)
                coef_top_k_idx_nom = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]
                temp_u_vec = np.zeros_like(u_vec)
                temp_u_vec[coef_top_k_idx_nom] = u_vec[coef_top_k_idx_nom]
                u_vec = temp_u_vec
            self.nominal_directions.append(u_vec)      
            random_vector = np.random.rand(len(u_vec))
            basis_matrix = np.vstack(self.nominal_directions)
            if len(self.nominal_directions) > 1:
                if len(self.nominal_directions) >= (len(u_vec) - 1):
                    selected_basis_idx = np.random.choice(len(self.nominal_directions), (len(u_vec) - 1), replace=False)
                    basis_matrix = basis_matrix[selected_basis_idx, :]
                z_vec = find_orthogonal_vector_to_mat(basis_matrix)
                #z_vec = find_orthogonal_vector_qr(basis_matrix)
            else:
                z_vec = find_orthogonal_vector(random_vector, u_vec)
            #z_vec = find_orthogonal_vector(random_vector, u_vec)
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
    
    def replicate_user_feedback_simplified_anom(self, args, itr=0, use_mahal=False, use_orig_mu=True, use_top_k=True):
        dists = []
        if use_orig_mu:
            mu = np.mean(self.original_data, axis=0)
        else:
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
        if use_top_k:
            print("Using Top K Anom")
            z_vec = np.zeros_like(u_vec)
            z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
        else:
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
    
    def get_user_feedback(self, args, itr=0, use_mahal=False, use_orig_mu=True, use_top_k=True, negate=False):
        use_data = self.data[self.masked_data, :]
        use_labels = self.labels[self.masked_data]
        print("Use Data Shape: ", use_data.shape)
        original_dists = []
        orig_mu = np.mean(self.original_data, axis=0)
        for i in range(self.original_data.shape[0]):
            a_i = self.original_data[i, :]
            curr_dist = mahalanobis_fast_uv(a_i, orig_mu, self.curr_dist_mat)
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
        plt.savefig('./figures/debugging_figs_true/dist_hist_true_{}_{}'.format(args.data, itr))
        plt.close()



        if use_orig_mu:
            mu = np.mean(self.original_data, axis=0)
        else:
            mu = np.mean(use_data, axis=0)
        anomaly_scores = calc_anomaly_scores(features=use_data, A_t=self.curr_dist_mat, mean_vec=mu)
        percentile_thresh = np.percentile(anomaly_scores, self.args.selection_thresh)
        percentile_thresh_median = np.percentile(anomaly_scores, 50)
        percentile_thresh_conservative = np.percentile(anomaly_scores, self.args.conserv_thresh)
        likely_anomalies = np.where(anomaly_scores > percentile_thresh)[0]
        # select the maximal anomaly score
        
        if self.args.prime_anomalies and itr < self.args.prime_limit:
            selected_idx = self.get_first_anomaly(anomaly_scores, use_data, use_labels)
        elif self.args.prime_nominals and itr < self.args.prime_limit:
            selected_idx = self.get_first_nominal(anomaly_scores, use_data, use_labels)
        elif self.args.use_thresh:
            print("* Using Threshold for Point Selection *".format(percentile_thresh))
            print("Percentile Thresh: {}".format(percentile_thresh))
            print("Percentile Thresh Cons: {}".format(percentile_thresh_conservative))
            # selected idx is the closest point greater than the threshold
            selected_idx = likely_anomalies[np.argmin(np.abs(anomaly_scores[likely_anomalies] - percentile_thresh))]
        else:
            # greedy selection
            selected_idx = np.argmax(anomaly_scores)
        
        selected_score = anomaly_scores[selected_idx]
        selected_point = use_data[selected_idx, :]
        selected_label = use_labels[selected_idx]
        print("Selected Point Score: {}".format(selected_score))
        print("Score Mean: {}".format(np.mean(anomaly_scores)))
        print("Score Max: {}".format(np.max(anomaly_scores)))
        print("Score Min: {}".format(np.min(anomaly_scores)))
        # likely anomalies are those outside the 99th percentile
        # likely nominal are those inside the 50th percentile
        likely_nominals = np.where(anomaly_scores < percentile_thresh_conservative)[0]
        a_1 = mu
        # if anomaly
        if selected_label == 0:
            print("Anomaly Found")
            self.num_anomalies_queried += 1
            self.queried_anomalies.append(selected_point)
            #a_2_idx = np.random.choice(likely_nominals, size=1)[0]
            #a_2 = self.data[a_2_idx, :]
            # select point closest to percentile_thresh, but less than it, from self.data
            likely_nominal_dists = anomaly_scores[likely_nominals]
            a_2_idx = np.argmin(np.abs(likely_nominal_dists - percentile_thresh))
            a_2_idx = likely_nominals[a_2_idx]
            a_2 = use_data[a_2_idx, :]
            a_k = selected_point

            diff = a_k - mu
            if use_mahal:
                mahal_grad = self.curr_dist_mat @ diff
            else:
                mahal_grad = diff
            u_vec = mahal_grad/np.linalg.norm(mahal_grad)
            coef_abs = np.abs(u_vec)
            coef_topk_idx = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]
            if use_top_k:
                print("Using Top K Anom")
                z_vec = np.zeros_like(u_vec)
                z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
            else:
                z_vec = u_vec

            z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
            z_vec = np.expand_dims(z_vec, 1)
            #print(z_vec)
            curr_Z = z_vec@z_vec.T

            
        # if nominal
        else:
            if len(self.queried_anomalies) > 0:
                queried_anomaly_scores = calc_anomaly_scores(features=np.array(self.queried_anomalies), A_t=self.curr_dist_mat, mean_vec=mu)
                # select the maximal anomaly score
                a_k_idx = np.argmax(queried_anomaly_scores)
                a_k = self.queried_anomalies[a_k_idx]
            else:
                #a_k_idx = np.random.choice(likely_anomalies, size=1)[0]
                #a_k = self.data[a_k_idx, :]
                # select the point with the second highest anomaly score
                a_k_idx = np.argsort(anomaly_scores)[::-1][1]
                a_k = use_data[a_k_idx, :]
            a_2 = selected_point

            if (self.args.use_oml and len(self.queried_anomalies) <= 0) or (self.args.oml_always):
                if len(self.queried_anomalies) > 0 and self.args.anom_pairwise:
                    ##########################
                    # THIS SECTION DOESN'T WORK WELL
                    self.oml_flag = True
                    queried_anomaly_scores = calc_anomaly_scores(features=np.array(self.queried_anomalies), A_t=self.curr_dist_mat, mean_vec=mu)
                    # select the maximal anomaly score
                    a_k_idx = np.argmax(queried_anomaly_scores)
                    a_k = self.queried_anomalies[a_k_idx]
                    curr_anom_nom_dist = mahalanobis_fast_uv(selected_point, a_k, self.curr_dist_mat)
                    diff_class_dist = curr_anom_nom_dist*2
                    print("Using OML Anom Pairwise: Dist {}".format(diff_class_dist))
                    #curr_Z = self.oml_branch(u_point=selected_point, mu=a_k, y_t=diff_class_dist)
                    print("Old Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, a_k, self.curr_dist_mat)))
                    self.oml_branch(u_point=selected_point, mu=a_k, y_t=diff_class_dist)
                    print("New Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, a_k, self.curr_dist_mat)))
                    ###########################
                else:
                    print("Using OML")
                    self.oml_flag = True
                    #curr_Z = self.oml_branch(u_point=selected_point, mu=mu)
                    print("Old Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, mu, self.curr_dist_mat)))
                    #self.oml_branch(u_point=selected_point, mu=mu, y_t=self.args.same_class_dist)
                    self.oml_branch(u_point=selected_point, mu=mu, y_t=percentile_thresh_median)
                    print("New Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, mu, self.curr_dist_mat)))

            elif self.args.nom_deweight:
                # don't build a weak learner, just deweight the features
                print("Deweighting Features")
                self.oml_flag = True # overload this flag
                print("Old Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, mu, self.curr_dist_mat)))
                self.deweight_features(u_point=selected_point, mean=mu)
                print("New Distance Between Points: {}".format(mahalanobis_fast_uv(selected_point, mu, self.curr_dist_mat)))
            else:
            #########
                self.queried_nominals.append(selected_point)

                diff = a_2 - mu
                if use_mahal:
                    mahal_grad = self.curr_dist_mat @ diff
                else:
                    mahal_grad = diff
                u_vec = mahal_grad/np.linalg.norm(mahal_grad)
                if use_top_k:
                    print("Using Top K Nom")
                    coef_abs = np.abs(u_vec)
                    coef_top_k_idx_nom = np.argpartition(coef_abs, -self.top_k)[-self.top_k:]
                    temp_u_vec = np.zeros_like(u_vec)
                    temp_u_vec[coef_top_k_idx_nom] = u_vec[coef_top_k_idx_nom]
                    u_vec = temp_u_vec
                self.nominal_directions.append(u_vec)      
                random_vector = np.random.rand(len(u_vec))
                basis_matrix = np.vstack(self.nominal_directions)
                if len(self.nominal_directions) > 1:
                    if len(self.nominal_directions) >= (len(u_vec) - 1):
                        #selected_basis_idx = np.random.choice(len(self.nominal_directions), (len(u_vec) - 1), replace=False)
                        #basis_matrix = basis_matrix[selected_basis_idx, :]
                        # remove the first index of nominal directions
                        self.nominal_directions = self.nominal_directions[1:]
                        basis_matrix = np.vstack(self.nominal_directions)
                    z_vec = find_orthogonal_vector_to_mat(basis_matrix)
                    #z_vec = find_orthogonal_vector_qr(basis_matrix)
                else:
                    z_vec = find_orthogonal_vector(random_vector, u_vec)
                #z_vec = find_orthogonal_vector(random_vector, u_vec)
                if negate:
                    z_vec = u_vec

                # construct weak learner from vector of most important features
                # z_vec = np.zeros_like(u_vec)
                # z_vec[coef_topk_idx] = u_vec[coef_topk_idx]
                z_vec = z_vec/(np.linalg.norm(z_vec) + 1e-12)
                z_vec = np.expand_dims(z_vec, 1)
                #print(z_vec)
                curr_Z = z_vec@z_vec.T
            ##########
        # remove selected point from data
        selected_indices = np.flatnonzero(self.masked_data)
        original_idx = selected_indices[selected_idx]
        print("Removing Point {}".format(original_idx))
        self.queried_indices.append(original_idx)
        self.masked_data[original_idx] = False
        #self.data = np.delete(self.data, selected_idx, axis=0)
        #self.labels = np.delete(self.labels, selected_idx, axis=0)

        curr_triplet = (a_1, a_2, a_k)
        if self.oml_flag: # I want to throw up this is horrible code lol
            curr_triplet = (None, None, None)
            curr_Z = None
        #print(selected_point)
        
        return None, curr_Z, curr_triplet


    def get_first_anomaly(self, anomaly_scores, use_data, use_labels):
        print("Selecting First Anomaly")
        # sort anomaly score indices in descending order
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        # get the first anomaly
        for idx in sorted_indices:
            if use_labels[idx] == 0: # anomaly is index 0 here
                return idx
    
        return None # no anomalies found
    
    def get_first_nominal(self, anomaly_scores, use_data, use_labels):
        print("Selecting First Nominal")
        # sort anomaly score indices in descending order
        sorted_indices = np.argsort(anomaly_scores)
        # get the first anomaly
        for idx in sorted_indices:
            if use_labels[idx] == 1: # nominal is index 1 here
                return idx
        
        return None # no nominals found
    
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
            inn_third = -w_j*H_s[r]
            inn_third = np.clip(inn_third, -100, 100)
            third = np.exp(inn_third) # overflow can happen here...
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
    
    def oml_branch(self, u_point, mu, y_t=10.0, learning_rate_factor=1.0):
        """
        OML Branching Algorithm
        
        Parameters
        ----------
        u_point : np.ndarray
        mu : np.ndarray
        y_t : float
            Desired distance
        """
        print("Y_t {}".format(y_t))
        u_t = np.expand_dims(u_point, 1)
        yhat_t = mahalanobis_fast_uv(u_t.flatten(), mu.flatten(), self.curr_dist_mat)
        ybar = calc_ybar(eta=self.args.eta, y_t=y_t, yhat_t=yhat_t)
        print("Ybar {}".format(ybar))
        #delta_update = self.calc_A_update(eta=self.args.eta, ybar=ybar, y_t=y_t, A_t=self.curr_dist_mat.copy(), u_t=u_t, v_t=np.expand_dims(mu, 1))
        A_tpo = calc_A_tpo(eta=self.args.eta, ybar=ybar, y_t=y_t, A_t=self.curr_dist_mat, u_t=u_t, v_t=np.expand_dims(mu, 1))
        # if self.args.use_top_k:
        #     print("Using top k OML")
        #     # get top k indices of diagonal
        #     diag_vals = np.abs(np.diag(delta_update))
        #     top_k_idx = np.argpartition(diag_vals, -self.top_k)[-self.top_k:]
        #     A_tpo_zeros = np.zeros_like(delta_update)
        #     # copy over i,j pairs of top k indices
        #     A_tpo_zeros[top_k_idx, top_k_idx] = delta_update[top_k_idx, top_k_idx]
        #     delta_update = A_tpo_zeros
        #     A_tpo = self.curr_dist_mat - delta_update
        #     print("Delta Update Max {} Min {}".format(delta_update.max(), delta_update.min()))
        self.curr_dist_mat = A_tpo
        #print(delta_update)
        #exit()
        #print("Learning Rate Factor {}".format(learning_rate_factor))
        #delta_update = delta_update*learning_rate_factor
        #print(delta_update)
        #delta_update = self.make_rank_one_trace_one(delta_update)
        #print(delta_update)
        #exit()
        #return delta_update

    def calc_A_update(self, eta, ybar, y_t, A_t, u_t, v_t):
        z_t = u_t - v_t
        first = eta*(ybar - y_t)
        second = A_t@z_t@z_t.T@A_t
        numer = first*second

        dsecond = z_t.T@A_t@z_t
        denom = 1 + first*dsecond

        #A_tpo = A_t - numer/denom
        
        return numer/denom
    
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

        
    
    def make_rank_one_trace_one(self, Z):
        # Step 1: Extract the dominant eigenvector of Z
        eigvals, eigvecs = np.linalg.eig(Z)
        dominant_eigenvector = eigvecs[:, np.argmax(eigvals.real)]  # Select eigenvector for the largest eigenvalue
        dominant_eigenvector = dominant_eigenvector.real  # Ensure it's real-valued (eigenvalues may have small imaginary parts due to numerical errors)
        
        # Step 2: Normalize the dominant eigenvector to satisfy the trace-one constraint
        normalized_vector = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)
        
        # Step 3: Create the rank-one, trace-one matrix
        rank_one_trace_one_matrix = np.outer(normalized_vector, normalized_vector)
        rank_one_trace_one_matrix /= np.trace(rank_one_trace_one_matrix)  # Ensure trace is one
        
        return rank_one_trace_one_matrix
    

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

def final_classification(features, labels, X):
    dists = []
    chisq_vals = []
    mu = np.mean(features, axis=0)
    for i in range(features.shape[0]):
        a_i = features[i, :]
        curr_dist = mahalanobis_fast_uv(a_i, mu, X)
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

def calc_anomaly_scores(features, A_t, mean_vec):
    dists = []
    for i in range(features.shape[0]):
        #v = np.expand_dims(features[i, :], 1)
        v = features[i, :]

        #curr_dist = np.sqrt(my_mahal_squared(mean_vec, v, A_t)[0][0])
        curr_dist = mahalanobis_fast_uv(v, mean_vec, A_t)
        dists.append(curr_dist)

    return np.array(dists)

def mahal_all_points(features, X):
    dist_mat = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            u = features[i, :]
            v = features[j, :]
            curr_dist = mahalanobis_fast_uv(u, v, X)
            dist_mat[i, j] = curr_dist
    
    return np.array(dist_mat)

def find_orthogonal_vector(v, reference):
    """Find an orthogonal vector to the given vector v using the reference vector."""
    v_orthogonal = v - (np.dot(v, reference) / np.dot(reference, reference)) * reference
    return v_orthogonal

def find_orthogonal_vector_to_mat(A):
    nullspace = null_space(A)
    
    if nullspace.size == 0:
        raise ValueError("No orthogonal vector exists; the input vectors span the entire space.")
    
    # Return one vector from the null space (e.g., the first column)
    orthogonal_vector = nullspace[:, 0]
    
    # Normalize the vector (optional)
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    for row in A:
        assert np.isclose(np.dot(row, orthogonal_vector), 0, atol=1e-10), \
            "Computed vector is not orthogonal to the input vectors."
    
    return orthogonal_vector

def find_orthogonal_vector_qr(matrix):
    """
    Finds a vector orthogonal to all rows of the given matrix using QR decomposition.

    Args:
        matrix (numpy.ndarray): A 2D array where each row is a vector.

    Returns:
        numpy.ndarray: A vector that is orthogonal to all rows of the input matrix.
    """
    # Perform QR decomposition on the transposed matrix
    Q, R = np.linalg.qr(matrix.T)
    
    # Find the dimension of the null space
    rank = np.linalg.matrix_rank(matrix)
    n = matrix.shape[1]
    
    if rank == n:
        raise ValueError("No orthogonal vector exists; the input vectors span the entire space.")
    
    # The null space corresponds to the last (n - rank) columns of Q
    orthogonal_vector = Q[:, rank]  # Take the first null space vector (column rank+1)
    
    # Normalize the vector (optional)
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    
    # Verify orthogonality
    for row in matrix:
        assert np.isclose(np.dot(row, orthogonal_vector), 0, atol=1e-10), \
            "Computed vector is not orthogonal to the input vectors."
    
    return orthogonal_vector