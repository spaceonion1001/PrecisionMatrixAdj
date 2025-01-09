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