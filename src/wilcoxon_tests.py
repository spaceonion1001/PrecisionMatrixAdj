import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from sklearn.metrics import auc
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score, pairwise_distances, roc_auc_score
from scipy.stats import wilcoxon
import re
import glob
import csv

from plotting import get_last_seed_f1, get_last_seed_ap, parse_f1_scores, parse_ap_scores, extract_f1_95, extract_avgprec_95, find_gmm_f1_file, parse_ap_scores_full, parse_f1_scores_full
sns.set()


base_dir_fif = ''
base_dir_aad = ''
base_dir_results = ''
base_dir_gaod = ''
base_dir_results_gmm = ''

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

def process_wilcoxon_results_fancy():
    specific_datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'imagenet', 'cifar_airplane', 'cifar_bird', 'fashion_sandal', 'fashion_boot', 'nslkdd', 'msl', 'mnist']
    cifar_datasets = ['CIFAR_{}'.format(x) for x in range(10)]
    fashion_datasets = ['Fashion_{}'.format(x) for x in range(10)]
    datasets = specific_datasets + cifar_datasets + fashion_datasets

    results_table = []

    for dataset in datasets:
        dataset_lower = dataset.lower()
        our_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_v2_nominal_deweight.txt')
        default_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_default.txt')
        gmm_f1_file = find_gmm_f1_file(base_dir_results_gmm, dataset_lower)
        gaod_f1_file = os.path.join(base_dir_gaod, f'{dataset_lower}.txt')

        curr_fif_f1 = [get_fif_scores(dataset_lower, it)[0] for it in range(10)]
        curr_fif_auc_pr = [get_fif_scores(dataset_lower, it)[1] for it in range(10)]

        curr_aad_if_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full.txt'))
        curr_aad_loda_f1 = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_f1_avg_full_loda.txt'))
        curr_aad_if_f1_full = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_fullf1_.txt'))
        curr_aad_loda_f1_full = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_fullf1__loda.txt'))
        curr_aad_if_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full.txt'))
        curr_aad_if_auc_pr_full = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_fullauc_.txt'))
        curr_aad_loda_auc_pr = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_auc_avg_precision_full_loda.txt'))
        curr_aad_loda_auc_pr_full = np.loadtxt(os.path.join(base_dir_aad, f'{dataset_lower}_fullauc__loda.txt'))

        our_f1_single = get_last_seed_f1(our_f1_file)
        default_f1 = get_last_seed_f1(default_f1_file)
        our_auc_pr_single = get_last_seed_ap(our_f1_file)
        default_auc_pr = get_last_seed_ap(default_f1_file)

        our_f1_single_full = np.array([our_f1_single] * 10)
        our_auc_pr_single_full = np.array([our_auc_pr_single] * 10)
        default_f1_full = np.array([default_f1] * 10)
        default_auc_pr_full = np.array([default_auc_pr] * 10)

        our_f1_gmm_full = parse_f1_scores_full(gmm_f1_file)
        our_auc_pr_gmm_full = parse_ap_scores_full(gmm_f1_file)
        gaod_f1_full = np.loadtxt(os.path.join(base_dir_gaod, f'{dataset_lower}_f1.txt'))
        gaod_auc_pr_full = np.loadtxt(os.path.join(base_dir_gaod, f'{dataset_lower}_avgprec.txt'))

        scores_f1 = {
            'FIF': np.mean(curr_fif_f1),
            'AAD_IF': curr_aad_if_f1,
            'AAD_LODA': curr_aad_loda_f1,
            'Ours': our_f1_single,
            'GAOD': extract_f1_95(gaod_f1_file),
            'GMM': np.mean(our_f1_gmm_full),
            'Default': default_f1
        }

        scores_ap = {
            'FIF': np.mean(curr_fif_auc_pr),
            'AAD_IF': curr_aad_if_auc_pr,
            'AAD_LODA': curr_aad_loda_auc_pr,
            'Ours': our_auc_pr_single,
            'GAOD': extract_avgprec_95(gaod_f1_file),
            'GMM': np.mean(our_auc_pr_gmm_full),
            'Default': default_auc_pr
        }

        full_scores_f1 = {
            'FIF': np.array(curr_fif_f1),
            'AAD_IF': curr_aad_if_f1_full,
            'AAD_LODA': curr_aad_loda_f1_full,
            'Ours': our_f1_single_full,
            'GAOD': gaod_f1_full,
            'GMM': our_f1_gmm_full,
            'Default': default_f1_full
        }

        full_scores_ap = {
            'FIF': np.array(curr_fif_auc_pr),
            'AAD_IF': curr_aad_if_auc_pr_full,
            'AAD_LODA': curr_aad_loda_auc_pr_full,
            'Ours': our_auc_pr_single_full,
            'GAOD': gaod_auc_pr_full,
            'GMM': our_auc_pr_gmm_full,
            'Default': default_auc_pr_full
        }

        sorted_f1 = sorted(scores_f1.items(), key=lambda x: x[1], reverse=True)
        sorted_ap = sorted(scores_ap.items(), key=lambda x: x[1], reverse=True)

        top1_f1_name, top2_f1_name = sorted_f1[0][0], sorted_f1[1][0]
        top1_ap_name, top2_ap_name = sorted_ap[0][0], sorted_ap[1][0]

        stat_f1, pval_f1 = wilcoxon(full_scores_f1[top1_f1_name], full_scores_f1[top2_f1_name], alternative='greater')
        stat_ap, pval_ap = wilcoxon(full_scores_ap[top1_ap_name], full_scores_ap[top2_ap_name], alternative='greater')
        print(full_scores_f1[top1_f1_name].shape, full_scores_f1[top2_f1_name].shape)
        print(full_scores_ap[top1_ap_name].shape, full_scores_ap[top2_ap_name].shape)
        print(f"    Wilcoxon F1: {stat_f1:.3f}, p-value: {pval_f1:.3f}")
        print(f"    Wilcoxon AP: {stat_ap:.3f}, p-value: {pval_ap:.3f}")

        results_table.append({
            "Dataset": dataset,
            "F1_p_value": round(pval_f1, 3),
            "F1_Top1": top1_f1_name,
            "F1_Top2": top2_f1_name,
            "AP_p_value": round(pval_ap, 3),
            "AP_Top1": top1_ap_name,
            "AP_Top2": top2_ap_name
        })

    # Save to CSV
    output_file = os.path.join("wilcoxon_summary_table.csv")
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = ["Dataset", "F1_p_value", "F1_Top1", "F1_Top2", "AP_p_value", "AP_Top1", "AP_Top2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_table:
            writer.writerow(row)

    print(f"✅ Wilcoxon summary table written to {output_file}")





if __name__ == "__main__":
    process_wilcoxon_results_fancy()
