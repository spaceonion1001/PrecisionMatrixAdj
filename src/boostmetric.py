import numpy as np
from sklearn.utils.extmath import fast_logdet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import normalize
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score
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
    else:
        print("Incorrect Dataset...")
        exit(1)


class BoostMetric:
    def __init__(self, data, labels, v, J, total_data, total_labels, keep_top_k=False, top_k=3):
        self.data = data
        self.labels = labels
        self.total_data = total_data
        self.total_labels = total_labels
        self.v = v
        self.J = J
        self.triplet_idxs = convert_to_triples(data, labels)
        print("Number of Triplets {}".format(len(self.triplet_idxs)))
        self.A_arr = calculate_AR(self.data, self.triplet_idxs)
        print("A length {}".format(self.A_arr.shape))

        self.u_r = np.ones(self.A_arr.shape[0])*(1/self.A_arr.shape[0])

        self.keep_top_k = keep_top_k
        self.top_k = top_k

    def iterate(self):
        w_s = []
        Z_s = []
        margins_count = []
        for j in range(self.J):
            A_hat_curr = self.calc_Ahat()
            if not is_symmetric(A_hat_curr):
                A_hat_curr = (A_hat_curr + A_hat_curr.T)/2
            #print("A_hat", A_hat_curr)
            eigvals, eigvecs = np.linalg.eigh(A_hat_curr)
            max_ind = np.argmax(eigvals)
            max_eig = eigvals[max_ind]
            max_vec = np.expand_dims(eigvecs[max_ind], 1)
            curr_Z = max_vec @ max_vec.T
            if not is_symmetric(curr_Z):
                curr_Z = (curr_Z + curr_Z.T)/2
            print("Iter {} Max Eig {}".format(j, max_eig))


            ###############################
            # keep only the top k features
            if self.keep_top_k:
                topk_ind = np.argpartition(np.diag(curr_Z), -self.top_k)[-self.top_k:]
                minimal_Z = np.zeros_like(curr_Z)
                np.fill_diagonal(minimal_Z, np.diag(curr_Z))
                for ind in topk_ind:
                    for ind2 in topk_ind:
                        minimal_Z[ind, ind2] = curr_Z[ind, ind2]
                curr_Z = minimal_Z.copy()
            ###############################

            #print("Eigvec", max_vec)
            #print(curr_Z)
            sns.heatmap(curr_Z)
            plt.savefig('figures/debugging_figs/Z_{}.png'.format(j))
            plt.close()
            if max_eig < self.v:
                print(">>> Max Eig {} Below Threshold - Exiting <<<".format(max_eig))
                break
            
            

            H_s = self.calc_H_r_js(curr_Z)
            w_j = self.bisection_search(curr_Z, H_s)
            # print("Most Violated Constraint Index {}".format(H_s.argmin()))
            # print("Most Violated Point Value {}".format(H_s[H_s.argmin()]))
            # print("Most Violated A {}".format(self.A_arr[H_s.argmin()]))
            # a_i = self.data[self.triplet_idxs[H_s.argmin()][0]]
            # a_j = self.data[self.triplet_idxs[H_s.argmin()][1]]
            # a_k = self.data[self.triplet_idxs[H_s.argmin()][2]]
            # dist_violated_ij = np.sqrt((a_i-a_j).T@(w_j*curr_Z)@(a_i-a_j))
            # dist_violated_ik = np.sqrt((a_i-a_k).T@(w_j*curr_Z)@(a_i-a_k))
            # dist_violated_ij_euclid = np.sqrt((a_i-a_j).T@(np.eye(curr_Z.shape[0]))@(a_i-a_j))
            # dist_violated_ik_euclid = np.sqrt((a_i-a_k).T@(np.eye(curr_Z.shape[0]))@(a_i-a_k))
            # print("Most Violated Dist IJ/IK(Z calc) {}/{}".format(dist_violated_ij, dist_violated_ik))
            # print("Most Violated Dist IJ/IK(Eye) {}/{}".format(dist_violated_ij_euclid, dist_violated_ik_euclid))
            # print("Feature Diff IJ {}".format(np.abs(a_i-a_j)))
            # print("Feature Diff IK {}".format(np.abs(a_i-a_k)))
            # print("EigVec {}".format(max_vec))
            # print("Eigvals {}".format(eigvals))
            # print("Argmax EigVal {}".format(eigvals.argmax()))
            # print("Argmin {}".format(H_s.argmin()))
            # print("Argmax Diff IJ {}".format(np.abs(a_i-a_j).argmax()))
            # print("Argmax Diff IK {}".format(np.abs(a_i-a_k).argmax()))
            # print("Argmax EigVec {}".format(max_vec.argmax()))
            #print("Most Violated Triplet {}".format(self.data[H_s.argmin(), :]))
            
            new_u = []
            for r in range(self.u_r.shape[0]):
                u_r_new = self.u_r[r]*np.exp(-H_s[r]*w_j)
                #print("Ur", u_r_new)
                new_u.append(u_r_new)
            new_u = np.array(new_u)
            new_u = new_u / new_u.sum()
            self.u_r = new_u
            #print("USum {}".format(self.u_r.sum()))
            #print("U", self.u_r)
            #print("H", H_s)
            w_s.append(w_j)
            Z_s.append(curr_Z)
            print("Curr Z Argmax {}".format(np.diag(curr_Z).argmax()))
            #print("W_j", w_j)
            #print("Z Shape", curr_Z.shape)
            curr_dist_mat = np.zeros((self.data.shape[1], self.data.shape[1]))
            for i in range(len(w_s)):
                curr_dist_mat += np.array(w_s)[i]*np.array(Z_s)[i]
            curr_margins = self.calc_H_r_js(curr_dist_mat)
            margins_count.append((curr_margins >= 0).sum()/len(curr_margins))

            sns.heatmap(curr_dist_mat)
            plt.savefig('figures/debugging_figs/X_{}.png'.format(j))
            plt.close()

            # knn
            #dmat = mahal_all_points(self.total_data, curr_dist_mat)
            #dist = DistanceMetric.get_metric('mahalanobis', VI=curr_dist_mat)
            #dmat = dist.pairwise(self.total_data, self.total_data)
            #dmat = pairwise_distances_chunked(X=self.total_data, Y=self.total_data, metric='mahalanobis', VI=curr_dist_mat)
            end_idx = 500
            total_iters = 0
            acc_scores = []
            f1_scores = []
            num_anomaly_labels = 0
            # shuffle data and labels together
            self.total_data, self.total_labels = shuffle(self.total_data, self.total_labels, random_state=12)
            while end_idx < 20000:
                print("> > Batch {} End Idx {} Last Idx {}".format(total_iters+1, end_idx, self.total_data.shape[0]))
                curr_tdata = self.total_data[end_idx-500:end_idx, :]
                curr_tlabels = self.total_labels[end_idx-500:end_idx]
                dmat = mahal_all_points(curr_tdata, curr_dist_mat)
                clf = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
                clf.fit(dmat, curr_tlabels)

                end_idx += 500
                total_iters += 1
                acc_scores.append(clf.score(dmat, curr_tlabels))
                f1_curr = f1_score(curr_tlabels, clf.predict(dmat))
                f1_scores.append(f1_curr)
                num_anomaly_labels += (curr_tlabels == 0).sum()
            print("KNN Score {}".format(sum(acc_scores)/total_iters))
            print("F1 Score {}".format(sum(f1_scores)/total_iters))
            print("Num Anomalies Checked {}".format(num_anomaly_labels))
           # print("KNN Score {}".format(clf.score(dmat, self.total_labels)))

        self.w_s = np.array(w_s)
        self.Z_s = np.array(Z_s)
        self.dist_mat = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(self.J):
            self.dist_mat += self.w_s[i]*self.Z_s[i]
        
        plt.plot(np.arange(len(margins_count)), margins_count)
        plt.xlabel("Iters")
        plt.ylabel("Satisfied Constraints")
        plt.ylim(0.0, 1.1)
        plt.savefig('./figures/margincounts.png')
        plt.close()
        return self.dist_mat
        

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
            

    def calc_H_r_js(self, curr_Z):
        H_s = []
        for r in range(self.A_arr.shape[0]):
            #print("A", self.A_arr[r].shape)
            #print("Z", curr_Z.shape)
            H_r_j = np.trace(self.A_arr[r] @ curr_Z.T)
            #print("H", H_r_j.shape)
            
            H_s.append(H_r_j)

        return np.array(H_s)
    
    def calc_lhs(self, H_s, w_j):
        summand = 0
        for r in range(self.A_arr.shape[0]):
            first = (H_s[r] - self.v)
            second = self.u_r[r]
            third = np.exp(-w_j*H_s[r]) # overflow can happen here...
            summand += first*second*third
        return summand
    
    def get_w_Z(self):
        return self.w_s, self.Z_s

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
        A_r = (A_r + A_r.T)/2
        assert is_symmetric(A_r)
        A_arr.append(A_r)
        
    return np.array(A_arr)

def init_covar(data):
    #A_t = np.cov(data.T)
    A_t = np.corrcoef(data.T)
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

def get_queries(anomaly_scores, features, labels, k=5):
    ind = np.argpartition(anomaly_scores, -k)[-k:]
    topk_scores = anomaly_scores[ind]
    topk_feats = features[ind]
    topk_labels = labels[ind]

    return ind, topk_feats, topk_labels

def main_boost(args):
    data, features, labels = resolve_data(args)
    features = remove_bad_features(features)
    print("{} Shape {}".format(args.data, features.shape))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    mean_vec = np.mean(features, axis=0)

    #anomaly_scores = calc_anomaly_scores(features=features, A_t=X, mean_vec=mean_vec)
    if args.data != 'wine':
        np.random.seed(42)
        data_pos = features[labels == 0]
        data_neg = features[labels == 1]
        pos_samples = np.random.choice(np.arange(data_pos.shape[0]), 10)
        neg_samples = np.random.choice(np.arange(data_neg.shape[0]), 100)
        pos_data = data_pos[pos_samples]
        pos_labels = np.zeros(pos_data.shape[0])
        neg_data = data_neg[neg_samples]
        neg_labels = np.ones(neg_data.shape[0])
        data_min = np.concatenate((pos_data, neg_data), 0)
        labels_min = np.concatenate((pos_labels, neg_labels))

        features = add_epsilon_noise(data_min)
        labels = labels_min
    #X = init_covar(features)
    X = np.eye(features.shape[1])
    bm = BoostMetric(data=features, labels=labels, v=args.v, J=args.iters)
    a_arr = bm.calc_H_r_js(curr_Z=X)
    pca = PCA(n_components=2)
    xt = pca.fit_transform(features)
    plotpca = plt.scatter(xt[:, 0], xt[:, 1], c=labels)
    plt.legend(handles=plotpca.legend_elements()[0], labels=[0, 1])
    plt.savefig('./figures/pca_{}.png'.format(args.data))
    plt.close()
    print((a_arr > 0.0).sum())
    print(a_arr)




def main(args):
    data, features, labels = resolve_data(args)
    features = remove_bad_features(features)
    print("{} Shape {}".format(args.data, features.shape))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    mean_vec = np.mean(features, axis=0)
    X = init_covar(features)
    anomaly_scores = calc_anomaly_scores(features=features, A_t=X, mean_vec=mean_vec)

    queried_feats = np.empty((0, features.shape[1]))
    queried_anomalies = np.empty((0, features.shape[1]))
    queried_labels = np.array([])
    num_queried = 0
    anoms_found = 0
    plotting_tups = []
    plotting_tups.append((0, 0))
    while num_queried < args.budget:
        topk_ind, topk_feats, topk_labels = get_queries(anomaly_scores=anomaly_scores, features=features, labels=labels, k=args.k)

        for i in range(topk_feats.shape[0]):
            curr_ind = topk_ind[i]
            curr_feat = topk_feats[i]
            curr_label = topk_labels[i]
            print("Index {} Label {}".format(curr_ind, curr_label))
            queried_feats = np.vstack((queried_feats, np.expand_dims(curr_feat, 0)))
            queried_labels = np.concatenate((queried_labels, np.expand_dims(curr_label, 0)))
            
            features = np.delete(features, curr_ind, axis=0)
            labels = np.delete(labels, curr_ind)
            if curr_label == 0:
                queried_anomalies = np.vstack((queried_anomalies, np.expand_dims(curr_feat, 0)))
                temp_features = features.copy()
                temp_features = np.vstack((queried_anomalies, temp_features))
                temp_labels = np.ones(len(labels))
                anom_labels = np.zeros(len(queried_anomalies))

                temp_labels = np.concatenate((anom_labels, temp_labels))

                clf = SelectKBest(f_classif, k=3).fit(temp_features, temp_labels)
                lda = LinearDiscriminantAnalysis()
                lda.fit(temp_features, temp_labels)
                scores = np.abs(clf.scores_)
                lda_coefs = lda.coef_.squeeze()
                lda_coefs[~clf.get_support()] = 0.0
                lda_coefs = lda_coefs / np.linalg.norm(lda_coefs)
                lda_coefs = np.expand_dims(lda_coefs, 1)
                Z = lda_coefs.dot(lda_coefs.T)
                X = X + Z

            anomaly_scores = calc_anomaly_scores(features=features, A_t=X, mean_vec=mean_vec)
            num_queried += 1
            topk_ind_int, topk_feats_int, topk_labels_int = get_queries(anomaly_scores=anomaly_scores, features=features, labels=labels, k=args.k)
            #print("Index New {} Label New {}".format(topk_ind_int, topk_labels_int))
            #curr_seen = seen_flags[topk_ind_int]
            num_anoms = (topk_labels_int == 0).sum() # 0 is the label for this data
            anoms_found += num_anoms
            plotting_tups.append((num_queried, anoms_found))
            if num_queried >= args.budget:
                break

    queries = np.array([x[0] for x in plotting_tups])
    detects = np.array([x[1] for x in plotting_tups])
    np.savetxt('results/{}/{}'.format(args.data, 'boostmet_detects_k{}.csv'.format(args.k)), detects.astype(int))
if __name__ == "__main__":
    args = parse_arguments()
    #main(args)
    #main_boost(args)

    data, features, labels = resolve_data(args)
    features = remove_bad_features(features)
    scaler = StandardScaler()
    #features = scaler.fit_transform(features)
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features, labels)
    print("Score {}".format(clf.score(features, labels)))
    print("Feature Importance {}".format(clf.feature_importances_))
    print("Feature Importance Argmax {}".format(clf.feature_importances_.argmax()))
    # clf = SVC(kernel='linear')
    # clf.fit(features, labels)
    # print("Score SVM {}".format(clf.score(features, labels)))
    # # print("Feature Importance SVM {}".format(clf.coef_[0]))
    # # print("Feature Importance SVM {}".format(clf.coef_[1]))
    # # print("Feature Importance SVM {}".format(clf.coef_[2]))
    # print("Feature Importance SVM {}".format(clf.coef_))
    # print("Feature Importance Argmax SVM {}".format(clf.coef_.argmax()))
    total_data = features.copy()
    total_labels = labels.copy()
    if args.data != 'wine':
        # data_pos = features[labels == 0]
        # data_neg = features[labels == 1]
        # pos_samples = np.random.choice(np.arange(data_pos.shape[0]), 10)
        # neg_samples = np.random.choice(np.arange(data_neg.shape[0]), 100)
        # pos_data = data_pos[pos_samples]
        # pos_labels = np.zeros(pos_data.shape[0])
        # neg_data = data_neg[neg_samples]
        # neg_labels = np.ones(neg_data.shape[0])
        # data_min = np.concatenate((pos_data, neg_data), 0)
        # labels_min = np.concatenate((pos_labels, neg_labels))

        #features = add_epsilon_noise(data_min)
        #labels = labels_min
        features, labels = smart_sampling(features=features, labels=labels, num_anoms=10, num_nominals=100)
        features = add_epsilon_noise(features=features)
    bm = BoostMetric(data=features, labels=labels, v=args.v, J=args.iters, total_data=total_data, total_labels=total_labels)
    X = bm.iterate()
    cws, cZs = bm.get_w_Z()
    print("Final ws", cws)
    print("X shape {}".format(X.shape))
    print(is_symmetric(X))
    print(is_pos_def(X))
    if not is_pos_def(X):
        X = X + 1e-8*np.eye(X.shape[0])
        print("Not PosDef EigVals {}".format(np.linalg.eigh(X)[0]))
    #print(X)
    #print(labels)
    # data[0] and data[1] have label 0
    # data[2], data[4] has label 1 
    # AT DIM=10
    # a_0 = np.expand_dims(features[0], 1)
    # a_1 = np.expand_dims(features[70], 1)
    # a_2 = np.expand_dims(features[-1], 1)
    # a_00 = np.expand_dims(features[1], 1)
    # dist_01 = np.sqrt((a_0-a_1).T@X@(a_0-a_1))
    # dist_02 = np.sqrt((a_0-a_2).T@X@(a_0-a_2))
    # dist_12 = np.sqrt((a_1-a_2).T@X@(a_1-a_2))
    # dist_00 = np.sqrt((a_0-a_00).T@X@(a_0-a_00))
    # print("Dist 0/1 {} Dist 0/2 {} Dist 1/2 {} Dist 00 {}".format(dist_01, dist_02, dist_12, dist_00))

    # XI = np.eye(features.shape[1])
    # dist_01 = np.sqrt((a_0-a_1).T@XI@(a_0-a_1))
    # dist_02 = np.sqrt((a_0-a_2).T@XI@(a_0-a_2))
    # dist_12 = np.sqrt((a_1-a_2).T@XI@(a_1-a_2))
    # dist_00 = np.sqrt((a_0-a_00).T@XI@(a_0-a_00))
    # print("EYE: Dist 0/1 {} Dist 0/2 {} Dist 1/2 {} Dist 00 {}".format(dist_01, dist_02, dist_12, dist_00))


# dim=10
# data, labels = sim_gmm(N=18, dim=dim)
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# print(data.shape, labels.shape)

# bm = BoostMetric(data=data, labels=labels, v=1e-5, J=100)
# X = bm.iterate()
# cws, cZs = bm.get_w_Z()
# print("Final ws", cws)
# print("X shape {}".format(X.shape))
# print(is_symmetric(X))
# print(is_pos_def(X))
# print(X)
# print(labels)

# # data[0] and data[1] have label 0
# # data[2], data[4] has label 1 
# # AT DIM=10
# a_0 = np.expand_dims(data[0], 1)
# a_1 = np.expand_dims(data[1], 1)
# a_2 = np.expand_dims(data[2], 1)
# a_4 = np.expand_dims(data[4], 1)
# dist_01 = np.sqrt((a_0-a_1).T@X@(a_0-a_1))
# dist_02 = np.sqrt((a_0-a_2).T@X@(a_0-a_2))
# dist_12 = np.sqrt((a_1-a_2).T@X@(a_1-a_2))
# dist_24 = np.sqrt((a_2-a_4).T@X@(a_2-a_4))
# print("Dist 0/1 {} Dist 0/2 {} Dist 1/2 {} Dist 2/4 {}".format(dist_01, dist_02, dist_12, dist_24))

# XI = np.eye(dim)
# dist_01 = np.sqrt((a_0-a_1).T@XI@(a_0-a_1))
# dist_02 = np.sqrt((a_0-a_2).T@XI@(a_0-a_2))
# dist_12 = np.sqrt((a_1-a_2).T@XI@(a_1-a_2))
# dist_24 = np.sqrt((a_2-a_4).T@XI@(a_2-a_4))
# print("EYE: Dist 0/1 {} Dist 0/2 {} Dist 1/2 {} Dist 2/4 {}".format(dist_01, dist_02, dist_12, dist_24))
#print("Final Zs(0-2) {} \n {} \n {}".format(cZs[0], cZs[1], cZs[2]))