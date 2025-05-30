import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from sklearn.metrics import auc
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances, roc_auc_score
import re
import glob

from plotting import get_last_seed_f1, get_last_seed_ap, parse_f1_scores, parse_ap_scores, extract_f1_95, extract_avgprec_95, find_gmm_f1_file
sns.set()

base_dir_fif = '/mnt/c/Users/xdink/Documents/Research/FeedbackIsolationForest-master/FeedbackIsolationForest-master/results/'
base_dir_aad = '/mnt/c/Users/xdink/Documents/Research/ad_examples-master/ad_examples-master/prec_results/'
base_dir_results = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results/'
base_dir_gaod = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_gaod/'
base_dir_results_gmm = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_gmms_seeds/'


def get_fif_scores(dataset, it):
    file_path = os.path.join(base_dir_fif, f'{dataset}_iter{it}_scores.csv')
    file = pd.read_csv(file_path, sep=',')
    index = file['index'].values
    label = file['label'].values
    scores = file['score'].values
    scores = -scores  # Invert scores for maxing
    # convert label nominal to 0, anomaly to 1
    label = np.where(label == 'nominal', 0, 1)
    sorted_indices = np.argsort(scores)[::-1]
    top_5_percent = int(0.05 * sorted_indices.shape[0])
    top_5_indices = sorted_indices[:top_5_percent]
    preds = np.zeros(sorted_indices.shape[0])
    preds[top_5_indices] = 1  # anomaly
    f1 = f1_score(label, preds)
    avg_prec = average_precision_score(label, scores)

    return f1, avg_prec

def process_fif_results():
    datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'imagenet', 'cifar_airplane', 'cifar_bird', 'fashion_sandal', 'fashion_boot', 'nslkdd', 'msl', 'mnist']

    for dataset in datasets:
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []
        avg_precisions = []
        for it in range(0, 10):
            file_path = os.path.join(base_dir_fif, f'{dataset}_iter{it}_scores.csv')
            file = pd.read_csv(file_path, sep=',')
            index = file['index'].values
            label = file['label'].values
            scores = file['score'].values
            scores = -scores  # Invert scores for maxing
            # convert label nominal to 0, anomaly to 1
            label = np.where(label == 'nominal', 0, 1)
            # calculate precision, recall, f1 score, roc auc, average precision
            # set predictions to threshold at top 5%
            sorted_indices = np.argsort(scores)[::-1]
            top_5_percent = int(0.05*sorted_indices.shape[0])
            top_5_indices = sorted_indices[:top_5_percent]
            # print("Size of Top 5 Percent {}".format(top_5_indices.shape))
            preds = np.zeros(sorted_indices.shape[0])
            preds[top_5_indices] = 1 # anomaly
            precision = precision_score(label, preds)
            recall = recall_score(label, preds)
            f1 = f1_score(label, preds)
            roc_auc = roc_auc_score(label, scores)
            avg_prec = average_precision_score(label, scores)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            roc_aucs.append(roc_auc)
            avg_precisions.append(avg_prec)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)
        roc_aucs = np.array(roc_aucs)
        avg_precisions = np.array(avg_precisions)
        print(f"> Dataset: {dataset}")
        print(f"    Precision: {np.mean(precisions):.4f}")
        print(f"    Recall: {np.mean(recalls):.4f}")
        print(f"    F1 Score: {np.mean(f1_scores):.4f}")
        print(f"    ROC AUC: {np.mean(roc_aucs):.4f}")
        print(f"    Average Precision: {np.mean(avg_precisions):.4f}")

def process_cifar_fashion_results():
    cifar_datasets = ['cifar_{}'.format(x) for x in range(0, 10)]
    fashion_datasets = ['fashion_{}'.format(x) for x in range(0, 10)]
    datasets = cifar_datasets + fashion_datasets
    f1s_fif = []
    f1s_aad = []
    f1s_ours = []
    f1s_gaod = []
    f1s_gmm = []
    f1s_default = []

    aucs_pr_fif = []
    aucs_pr_aad = []
    aucs_pr_ours = []
    aucs_pr_gaod = []
    aucs_pr_gmm = []
    aucs_pr_default = []

    for dataset in datasets:
        our_f1_file = os.path.join(base_dir_results, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        default_f1_file = os.path.join(base_dir_results, 'precision_scores_{}_default.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_results_gmm, dataset)
        gaod_f1_file = os.path.join(base_dir_gaod, '{}.txt'.format(dataset))
        curr_fif_f1 = []
        curr_fif_auc_pr = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()

        curr_aad_if_f1 = np.loadtxt(os.path.join(base_dir_aad, '{}_f1_avg_full.txt'.format(dataset)))
        curr_aad_loda_f1 = np.loadtxt(os.path.join(base_dir_aad, '{}_f1_avg_full_loda.txt'.format(dataset)))
        curr_aad_if_auc_pr = np.loadtxt(os.path.join(base_dir_aad, '{}_auc_avg_precision_full.txt'.format(dataset)))
        curr_aad_loda_auc_pr = np.loadtxt(os.path.join(base_dir_aad, '{}_auc_avg_precision_full_loda.txt'.format(dataset)))

        our_f1_single = get_last_seed_f1(our_f1_file)
        default_f1 = get_last_seed_f1(default_f1_file)
        our_auc_pr_single = get_last_seed_ap(our_f1_file)
        default_auc_pr = get_last_seed_ap(default_f1_file)
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        our_auc_pr_gmm, _ = parse_ap_scores(gmm_f1_file)
        gaod_f1 = extract_f1_95(gaod_f1_file)
        gaod_auc_pr = extract_avgprec_95(gaod_f1_file)

        print(f"> Dataset: {dataset}")
        print(f"    FIF F1: {curr_fif_f1:.4f}, AUC PR: {curr_fif_auc_pr:.4f}")
        print(f"    AAD IF F1: {curr_aad_if_f1:.4f}, AUC PR: {curr_aad_if_auc_pr:.4f}")
        print(f"    AAD LODA F1: {curr_aad_loda_f1:.4f}, AUC PR: {curr_aad_loda_auc_pr:.4f}")
        print(f"    Ours F1: {our_f1_single:.4f}, AUC PR: {our_auc_pr_single:.4f}")
        print(f"    GAOD F1: {gaod_f1:.4f}, AUC PR: {gaod_auc_pr:.4f}")
        print(f"    GMM F1: {our_f1_gmm:.4f}, AUC PR: {our_auc_pr_gmm:.4f}")
        print(f"    Default F1: {default_f1:.4f}, AUC PR: {default_auc_pr:.4f}")


# def process_cifar_fashion_results_fancy():
#     cifar_datasets = ['CIFAR_{}'.format(x) for x in range(10)]
#     fashion_datasets = ['Fashion_{}'.format(x) for x in range(10)]
#     datasets = cifar_datasets + fashion_datasets

#     f1_rows = []
#     auc_pr_rows = []

#     for dataset in datasets:
#         dataset_lower = dataset.lower()
#         our_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_v2_nominal_deweight.txt')
#         default_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_default.txt')
#         gmm_f1_file = find_gmm_f1_file(base_dir_results_gmm, dataset_lower)
#         gaod_f1_file = os.path.join(base_dir_gaod, f'{dataset_lower}.txt')

#         curr_fif_f1 = []
#         curr_fif_auc_pr = []
#         for it in range(10):
#             f1, avg_prec = get_fif_scores(dataset_lower, it)
#             curr_fif_f1.append(f1)
#             curr_fif_auc_pr.append(avg_prec)
#         curr_fif_f1 = np.mean(curr_fif_f1)
#         curr_fif_auc_pr = np.mean(curr_fif_auc_pr)

#         curr_aad_if_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full.txt'))
#         curr_aad_loda_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full_loda.txt'))
#         curr_aad_if_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full.txt'))
#         curr_aad_loda_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full_loda.txt'))

#         our_f1_single = get_last_seed_f1(our_f1_file)
#         default_f1 = get_last_seed_f1(default_f1_file)
#         our_auc_pr_single = get_last_seed_ap(our_f1_file)
#         default_auc_pr = get_last_seed_ap(default_f1_file)
#         our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
#         our_auc_pr_gmm, _ = parse_ap_scores(gmm_f1_file)
#         gaod_f1 = extract_f1_95(gaod_f1_file)
#         gaod_auc_pr = extract_avgprec_95(gaod_f1_file)

#         print(f"> Dataset: {dataset}")
#         print(f"    FIF F1: {curr_fif_f1:.4f}, AUC PR: {curr_fif_auc_pr:.4f}")
#         print(f"    AAD IF F1: {curr_aad_if_f1:.4f}, AUC PR: {curr_aad_if_auc_pr:.4f}")
#         print(f"    AAD LODA F1: {curr_aad_loda_f1:.4f}, AUC PR: {curr_aad_loda_auc_pr:.4f}")
#         print(f"    Ours F1: {our_f1_single:.4f}, AUC PR: {our_auc_pr_single:.4f}")
#         print(f"    GAOD F1: {gaod_f1:.4f}, AUC PR: {gaod_auc_pr:.4f}")
#         print(f"    GMM F1: {our_f1_gmm:.4f}, AUC PR: {our_auc_pr_gmm:.4f}")
#         print(f"    Default F1: {default_f1:.4f}, AUC PR: {default_auc_pr:.4f}")

#         # Corrected LaTeX table row order
#         f1_row = f"{dataset} & {curr_aad_if_f1:.3f} & {curr_aad_loda_f1:.3f} & {curr_fif_f1:.3f} & {gaod_f1:.3f} & {default_f1:.3f} & {our_f1_single:.3f} & {our_f1_gmm:.3f} \\\\"
#         auc_row = f"{dataset} & {curr_aad_if_auc_pr:.3f} & {curr_aad_loda_auc_pr:.3f} & {curr_fif_auc_pr:.3f} & {gaod_auc_pr:.3f} & {default_auc_pr:.3f} & {our_auc_pr_single:.3f} & {our_auc_pr_gmm:.3f} \\\\"

#         f1_rows.append(f1_row)
#         auc_pr_rows.append(auc_row)

#     with open('f1_table_rows.txt', 'w') as f:
#         for row in f1_rows:
#             f.write(row + '\n')

#     with open('auc_pr_table_rows.txt', 'w') as f:
#         for row in auc_pr_rows:
#             f.write(row + '\n')

#     print("Saved F1 and AUC PR rows for LaTeX tables.")

def process_cifar_fashion_results_fancy():
    cifar_datasets = ['CIFAR_{}'.format(x) for x in range(10)]
    fashion_datasets = ['Fashion_{}'.format(x) for x in range(10)]
    datasets = cifar_datasets + fashion_datasets

    f1_rows = []
    auc_pr_rows = []

    for dataset in datasets:
        dataset_lower = dataset.lower()
        our_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_v2_nominal_deweight.txt')
        default_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_default.txt')
        gmm_f1_file = find_gmm_f1_file(base_dir_results_gmm, dataset_lower)
        gaod_f1_file = os.path.join(base_dir_gaod, f'{dataset_lower}.txt')

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        for it in range(10):
            f1, avg_prec = get_fif_scores(dataset_lower, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
        curr_fif_f1 = np.mean(curr_fif_f1)
        curr_fif_auc_pr = np.mean(curr_fif_auc_pr)

        curr_aad_if_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full.txt'))
        curr_aad_loda_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full_loda.txt'))
        curr_aad_if_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full.txt'))
        curr_aad_loda_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full_loda.txt'))

        our_f1_single = get_last_seed_f1(our_f1_file)
        default_f1 = get_last_seed_f1(default_f1_file)
        our_auc_pr_single = get_last_seed_ap(our_f1_file)
        default_auc_pr = get_last_seed_ap(default_f1_file)
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        our_auc_pr_gmm, _ = parse_ap_scores(gmm_f1_file)
        gaod_f1 = extract_f1_95(gaod_f1_file)
        gaod_auc_pr = extract_avgprec_95(gaod_f1_file)

        print(f"> Dataset: {dataset}")
        print(f"    FIF F1: {curr_fif_f1:.4f}, AUC PR: {curr_fif_auc_pr:.4f}")
        print(f"    AAD IF F1: {curr_aad_if_f1:.4f}, AUC PR: {curr_aad_if_auc_pr:.4f}")
        print(f"    AAD LODA F1: {curr_aad_loda_f1:.4f}, AUC PR: {curr_aad_loda_auc_pr:.4f}")
        print(f"    Ours F1: {our_f1_single:.4f}, AUC PR: {our_auc_pr_single:.4f}")
        print(f"    GAOD F1: {gaod_f1:.4f}, AUC PR: {gaod_auc_pr:.4f}")
        print(f"    GMM F1: {our_f1_gmm:.4f}, AUC PR: {our_auc_pr_gmm:.4f}")
        print(f"    Default F1: {default_f1:.4f}, AUC PR: {default_auc_pr:.4f}")
        f1_vals = [
            curr_aad_if_f1,
            curr_aad_loda_f1,
            curr_fif_f1,
            gaod_f1,
            default_f1,
            our_f1_single,
            our_f1_gmm
        ]
        auc_vals = [
            curr_aad_if_auc_pr,
            curr_aad_loda_auc_pr,
            curr_fif_auc_pr,
            gaod_auc_pr,
            default_auc_pr,
            our_auc_pr_single,
            our_auc_pr_gmm
        ]

        max_f1 = max(f1_vals)
        max_auc = max(auc_vals)

        f1_row = f"{dataset} & " + " & ".join([
            f"\\textbf{{{v:.3f}}}" if v == max_f1 else f"{v:.3f}"
            for v in f1_vals
        ]) + " \\\\"
        auc_row = f"{dataset} & " + " & ".join([
            f"\\textbf{{{v:.3f}}}" if v == max_auc else f"{v:.3f}"
            for v in auc_vals
        ]) + " \\\\"

        f1_rows.append(f1_row)
        auc_pr_rows.append(auc_row)

    with open('f1_table_rows.txt', 'w') as f:
        for row in f1_rows:
            f.write(row + '\n')

    with open('auc_pr_table_rows.txt', 'w') as f:
        for row in auc_pr_rows:
            f.write(row + '\n')

    print("Saved F1 and AUC PR rows for LaTeX tables.")









            
            


if __name__ == "__main__":
    #process_fif_results()
    # process_cifar_fashion_results()
    process_cifar_fashion_results_fancy()
    