import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from sklearn.metrics import auc
import re
import glob
from sklearn.metrics import f1_score, average_precision_score
sns.set()

# # Use LaTeX-style fonts for publications
# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],  # or "Computer Modern Roman"
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 11,
#     "pdf.fonttype": 42,  # TrueType fonts for vector export
#     "ps.fonttype": 42
# })

base_dir_aad = ''
base_dir_boost = ''
save_dir = ''
base_dir_fif = ''

marker_size = 200
ref_font_size = 22
tick_label_size = 16
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

def extract_runtime(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "runtime" in line.lower():
                parts = line.strip().split(":")
                if len(parts) == 2:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        pass
    raise ValueError("Runtime not found or could not be parsed.")

def plot_aad_vs_boost(aad_file, boost_file, save_path, dataset):
    aad_np = np.loadtxt(os.path.join(base_dir_aad, aad_file), delimiter=',')[:, 2:].mean(axis=0)
    boost_np = np.loadtxt(os.path.join(base_dir_boost, boost_file), delimiter=',')
    print(aad_np.shape, boost_np.shape)

    plotting_range = np.arange(1, len(aad_np) + 1)
    plt.figure(figsize=(15, 10))
    plt.plot(plotting_range, aad_np, '-o', label='AAD', color='blue')
    plt.plot(plotting_range, boost_np, '-o', label='Boost', color='red')
    plt.title(f'{dataset} Anomaly Discovery Curve', fontsize=24)
    plt.xlabel('Number of Queries', fontsize=20)
    plt.ylabel("Number of Anomalies Discovered (Cumulative)", fontsize=20)
    plt.xlim(-1, len(aad_np) + 1)
    plt.ylim(-1, len(aad_np) + 1)
    plt.legend(loc='best', fontsize=18)
    plt.savefig(os.path.join(save_dir, save_path))
    plt.close()

def main_plot_diff_k():
    datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'oxford', 'nslkdd']
    for dataset in datasets:
        aad_file = '{}/aad_if_50/num_seen_iforest.csv'.format(dataset)
        aad_np = np.loadtxt(os.path.join(base_dir_aad, aad_file), delimiter=',')[:, 2:].mean(axis=0)
        plotting_range = np.arange(1, len(aad_np) + 1)
        plt.figure(figsize=(15, 10))
        plt.plot(plotting_range, aad_np, '-o', label='AAD', color='blue')
        colors = ['red', 'green', 'purple', 'orange', 'black', 'brown']
        for i, k in enumerate([3, 5, 10, 15, 20]):
            boost_file = 'num_anomalies_queried_{}_normed_nok_k{}_deweight.csv'.format(dataset, k)
            boost_np = np.loadtxt(os.path.join(base_dir_boost, boost_file), delimiter=',')
            plt.plot(plotting_range, boost_np, '-o', label=f'Boost k={k}', color=colors[i])
        boost_full = np.loadtxt(os.path.join(base_dir_boost, 'num_anomalies_queried_{}_normed_nok_full_deweight.csv'.format(dataset)), delimiter=',')
        plt.plot(plotting_range, boost_np, '-o', label=f'Boost k=Full', color=colors[-1])
        plt.title(f'{dataset} Anomaly Discovery Curve', fontsize=24)
        plt.xlabel('Number of Queries', fontsize=20)
        plt.ylabel("Number of Anomalies Discovered (Cumulative)", fontsize=20)
        plt.xlim(-1, len(aad_np) + 1)
        plt.ylim(-1, len(aad_np) + 1)
        plt.legend(loc='best', fontsize=18)
        save_path = '{}_anomaly_curve_ks.png'.format(dataset)
        plt.savefig(os.path.join(save_dir, save_path))

def main(save_suffix='normed_nok_deweight'):
    datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'oxford', 'nslkdd']
    for dataset in datasets:
        aad_file = '{}/aad_if_50/num_seen_iforest.csv'.format(dataset)
        boost_file = 'num_anomalies_queried_{}_{}.csv'.format(dataset, save_suffix)
        save_path = '{}_{}_anomaly_curve.png'.format(dataset, save_suffix)
        plot_aad_vs_boost(aad_file, boost_file, save_path, dataset)


def get_last_seed_f1(file_path):
    # THIS IS BECAUSE OF APPEND INSTEAD OF WRITE
    # AND IT APPENDED THE LAST LINE TO THE FILE
    # THIS IS USED FOR PMA - DETERMINISTIC
    pattern = re.compile(r"Seed (\d+) F1 ([0-9.]+)")
    last_match = None

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last_match = (int(match.group(1)), float(match.group(2)))

    if last_match:
        seed, f1 = last_match
        #print(f"Last Seed {seed}: F1 = {f1:.4f}")
    else:
        print("No F1 score found.")

    return f1

def parse_f1_scores(file_path):
    f1_scores = []

    pattern = re.compile(r"Seed (\d+) F1 ([0-9.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                seed = int(match.group(1))
                f1 = float(match.group(2))
                f1_scores.append((seed, f1))

    # Sort by seed for consistent output
    f1_scores.sort(key=lambda x: x[0])

    #for seed, f1 in f1_scores:
        #print(f"Seed {seed}: F1 = {f1:.4f}")

    f1_values = np.array([f1 for _, f1 in f1_scores])
    mean_f1 = f1_values.mean()
    std_f1 = f1_values.std()

    #print("\n--- Overall Summary ---")
    #print(f"Average F1 over {len(f1_scores)} seeds: {mean_f1:.4f} ± {std_f1:.4f}")

    return mean_f1, std_f1

def parse_f1_scores_full(file_path):
    f1_scores = []

    pattern = re.compile(r"Seed (\d+) F1 ([0-9.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                seed = int(match.group(1))
                f1 = float(match.group(2))
                f1_scores.append((seed, f1))

    # Sort by seed for consistent output
    f1_scores.sort(key=lambda x: x[0])

    #for seed, f1 in f1_scores:
        #print(f"Seed {seed}: F1 = {f1:.4f}")

    f1_values = np.array([f1 for _, f1 in f1_scores])
    mean_f1 = f1_values.mean()
    std_f1 = f1_values.std()

    #print("\n--- Overall Summary ---")
    #print(f"Average F1 over {len(f1_scores)} seeds: {mean_f1:.4f} ± {std_f1:.4f}")

    return f1_values

def get_last_seed_ap(file_path):
    # Extract the last "Seed X AP Y.YYY" line
    pattern = re.compile(r"Seed (\d+) Average Precision ([0-9.]+)")
    last_match = None

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last_match = (int(match.group(1)), float(match.group(2)))

    if last_match:
        seed, ap = last_match
    else:
        print("No Average Precision (AP) score found.")
        ap = None

    return ap


def parse_ap_scores(file_path):
    ap_scores = []
    pattern = re.compile(r"Seed (\d+) Avg Prec ([0-9.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                seed = int(match.group(1))
                ap = float(match.group(2))
                ap_scores.append((seed, ap))

    # Sort by seed for consistent output
    ap_scores.sort(key=lambda x: x[0])

    ap_values = np.array([ap for _, ap in ap_scores])
    mean_ap = ap_values.mean()
    std_ap = ap_values.std()

    return mean_ap, std_ap

def parse_ap_scores_full(file_path):
    ap_scores = []
    pattern = re.compile(r"Seed (\d+) Avg Prec ([0-9.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                seed = int(match.group(1))
                ap = float(match.group(2))
                ap_scores.append((seed, ap))

    # Sort by seed for consistent output
    ap_scores.sort(key=lambda x: x[0])

    ap_values = np.array([ap for _, ap in ap_scores])
    mean_ap = ap_values.mean()
    std_ap = ap_values.std()

    return ap_values

def extract_f1_95(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "F1 95" in line:
                parts = line.strip().split()
                try:
                    f1_index = parts.index("F1") + 2
                    f1_value = float(parts[f1_index])
                    #print(f"F1 95 Score: {f1_value:.4f}")
                    return f1_value
                except (ValueError, IndexError):
                    print("Error parsing F1 95 value.")
                    return None
    print("No 'F1 95' line found.")
    return None

def extract_avgprec_95(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "AvgPrec 95" in line:
                parts = line.strip().split()
                try:
                    # Get the value after "AvgPrec" (assumes format: "AvgPrec 95: <value>")
                    avgprec_index = parts.index("AvgPrec") + 2
                    avgprec_value = float(parts[avgprec_index])
                    # print(f"AvgPrec 95 Score: {avgprec_value:.4f}")
                    return avgprec_value
                except (ValueError, IndexError):
                    print("Error parsing AvgPrec 95 value.")
                    return None
    print("No 'AvgPrec 95' line found.")
    return None


def find_gmm_f1_file(base_dir_gmm_f1, dataset):
    pattern = os.path.join(base_dir_gmm_f1, f'precision_scores_{dataset}_v2_nominal_deweight_*.txt')
    matches = glob.glob(pattern)

    # Filter out any file that contains 'default'
    matches = [m for m in matches if 'default' not in m]

    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching pattern: {pattern}")
    elif len(matches) > 1:
        raise ValueError(f"Multiple files matched for dataset '{dataset}':\n" + "\n".join(matches))
    return matches[0]

def main_plot_runtimes_vs_f1():
    
    datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'nslkdd', 'imagenet', 'cifar_airplane', 'cifar_bird', 'fashion_sandal', 'fashion_boot', 'nslkdd', 'msl', 'mnist']
    cifar_min_datasets = ['cifar', 'cifar_airplane', 'cifar_bird']
    fashion_min_datasets = ['fashion', 'fashion_sandal', 'fashion_boot']
    cifar_datasets = ['cifar_{}'.format(x) for x in range(10)]
    fashion_datasets = ['fashion_{}'.format(x) for x in range(10)]
    datasets = datasets + cifar_datasets + fashion_datasets
    base_dir_single_rt = ''
    base_dir_gmm_rt = ''
    base_dir_single_f1 = ''
    base_dir_gmm_f1 = ''
    gaod_base_dir = ''
    aad_base_dir = ''

    for dataset in datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        fig, ax = plt.subplots(figsize=(8.5, 6))

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']


        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            ax.scatter(xs, ys, color=colors[i], label=method,
                    marker=markers[i], s=marker_size+50, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

        if dataset == 'bank':
            dataset = 'bank_v2'
        elif dataset == 'campaign':
            dataset = 'bank_v1'
        ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
        ax.set_ylabel("F1 Score", fontsize=ref_font_size)
        if dataset == 'cifar':
            ax.set_title("F1 vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
        elif dataset == 'fashion':
            ax.set_title("F1 vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
        else:
            ax.set_title("F1 vs Runtime - {}".format(dataset), fontsize=ref_font_size+4)
        #ax.set_xscale("log")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        if dataset == 'bank_v1':
            ax.legend(loc='upper left', frameon=True, fontsize=ref_font_size-2)
        # ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)

            
        os.makedirs('rt_plots', exist_ok=True)
        plt.ylim(0.0, 1.05)
        ax.tick_params(axis='both', labelsize=tick_label_size)
        plt.savefig('rt_plots/{}_f1_vs_runtime.png'.format(dataset), format='png', dpi=300)
        plt.show()
        plt.close()
    

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted_labels = set()

    for dataset in cifar_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("F1 Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("F1 vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("F1 vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("F1 vs Runtime - CIFAR-10".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)

            
    os.makedirs('rt_plots', exist_ok=True)
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/cifar_f1_runtime_grouped.png'.format(dataset), format='png', dpi=300)
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted_labels = set()

    for dataset in fashion_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("F1 Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("F1 vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("F1 vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("F1 vs Runtime - FashionMNIST".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)

            
    os.makedirs('rt_plots', exist_ok=True)
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/fashion_f1_runtime_grouped.png'.format(dataset), format='png', dpi=300)
    plt.show()
    plt.close()


def main_plot_runtimes_vs_prauc():
    datasets = ['bank', 'campaign', 'unsw', 'cifar', 'fashion', 'nslkdd', 'imagenet', 'cifar_airplane', 'cifar_bird', 'fashion_sandal', 'fashion_boot', 'nslkdd', 'msl', 'mnist']
    cifar_min_datasets = ['cifar', 'cifar_airplane', 'cifar_bird']
    fashion_min_datasets = ['fashion', 'fashion_sandal', 'fashion_boot']
    cifar_datasets = ['cifar_{}'.format(x) for x in range(10)]
    fashion_datasets = ['fashion_{}'.format(x) for x in range(10)]
    datasets = datasets + cifar_datasets + fashion_datasets
    base_dir_single_rt = ''
    base_dir_gmm_rt = ''
    base_dir_single_f1 = ''
    base_dir_gmm_f1 = ''
    gaod_base_dir = ''
    aad_base_dir = ''

    for dataset in datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_auc_avg_precision_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_auc_avg_precision_full_loda.txt'.format(dataset))

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_ap(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_ap_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_avgprec_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, AUC PR: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, AUC PR: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, AUC PR: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, AUC PR: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, AUC PR: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, AUC PR: {curr_fif_auc_pr:.4f}")

        
        # --- Plot: One point per method per dataset (outlier-style) ---
        fig, ax = plt.subplots(figsize=(8.5, 6))

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_auc_pr]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            ax.scatter(xs, ys, color=colors[i], label=method,
                    marker=markers[i], s=marker_size+50, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')
        if dataset == 'bank':
            dataset = 'bank_v2'
        elif dataset == 'campaign':
            dataset = 'bank_v1'
        ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
        ax.set_ylabel("AUPRC Score", fontsize=ref_font_size)
        if dataset == 'cifar':
            ax.set_title("AUPRC vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
        elif dataset == 'fashion':
            ax.set_title("AUPRC vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
        else:
            ax.set_title("AUPRC vs Runtime - {}".format(dataset), fontsize=ref_font_size+4)
        #ax.set_xscale("log")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        if dataset == 'bank_v1':
            ax.legend(loc='upper left', frameon=True, fontsize=ref_font_size-2)

            
        os.makedirs('rt_plots', exist_ok=True)
        plt.ylim(0.0, 1.05)
        ax.tick_params(axis='both', labelsize=tick_label_size)
        plt.savefig('rt_plots/{}_aucpr_vs_runtime.png'.format(dataset), format='png', dpi=300)
        plt.show()
        plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_labels = set()

    for dataset in cifar_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("AUPRC Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("AUPRC vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("AUPRC vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("AUPRC vs Runtime - CIFAR-10".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    #ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )


            
    os.makedirs('rt_plots', exist_ok=True)
    plt.tight_layout()
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/cifar_aucpr_runtime_grouped.png'.format(dataset), format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_labels = set()

    for dataset in fashion_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("AUPRC Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("AUPRC vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("AUPRC vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("AUPRC vs Runtime - FashionMNIST".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    # ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )
            
    os.makedirs('rt_plots', exist_ok=True)
    plt.tight_layout()
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/fashion_aucpr_runtime_grouped.png'.format(dataset), format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



    #############################################

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted_labels = set()

    for dataset in cifar_min_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("AUPRC Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("AUPRC vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("AUPRC vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("AUPRC vs Runtime - CIFAR-10 Rest-vs-One".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    # ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)

            
    os.makedirs('rt_plots', exist_ok=True)
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/cifar_aucpr_runtime_grouped_ovr.png'.format(dataset), format='png', dpi=300)
    plt.show()
    plt.close()

    #####################################

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted_labels = set()

    for dataset in fashion_min_datasets:
        runtimes = {}
        f1s = {}
        our_rt_file = os.path.join(base_dir_single_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        our_f1_file = os.path.join(base_dir_single_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_rt_file = os.path.join(base_dir_gmm_rt, 'runtimes_{}_v2_nominal_deweight.txt'.format(dataset))
        #gmm_f1_file = os.path.join(base_dir_gmm_f1, 'precision_scores_{}_v2_nominal_deweight.txt'.format(dataset))
        gmm_f1_file = find_gmm_f1_file(base_dir_gmm_f1, dataset)
        gaod_rt_file = os.path.join(gaod_base_dir, '{}_runtimes.txt'.format(dataset))
        gaod_f1_file = os.path.join(gaod_base_dir, '{}.txt'.format(dataset))
        aad_if_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full.txt'.format(dataset))
        aad_if_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full.txt'.format(dataset))
        aad_loda_rt_file = os.path.join(aad_base_dir, '{}_runtimes_full_loda.txt'.format(dataset))
        aad_loda_f1_file = os.path.join(aad_base_dir, '{}_f1_avg_full_loda.txt'.format(dataset))

        curr_fif_f1 = []
        curr_fif_auc_pr = []
        runtimes_fif = []
        for it in range(0, 10):
            f1, avg_prec = get_fif_scores(dataset, it)
            curr_fif_f1.append(f1)
            curr_fif_auc_pr.append(avg_prec)
            runtime_fif_path = os.path.join(base_dir_fif, 'runtime_iter{}_{}.txt'.format(it, dataset))
            runtimes_fif.append(extract_runtime(runtime_fif_path))
        curr_fif_f1 = np.array(curr_fif_f1).mean()
        curr_fif_auc_pr = np.array(curr_fif_auc_pr).mean()
        runtimes_fif = np.array(runtimes_fif).mean()

        our_runtime_single = np.loadtxt(our_rt_file, delimiter=',').mean()
        our_f1_single = get_last_seed_f1(our_f1_file)
        our_runtime_gmm = np.loadtxt(gmm_rt_file, delimiter=',').mean()
        our_f1_gmm, _ = parse_f1_scores(gmm_f1_file)
        gaod_runtime = np.loadtxt(gaod_rt_file, delimiter=',').mean()
        gaod_f1 = extract_f1_95(gaod_f1_file)
        aad_if_runtime = np.loadtxt(aad_if_rt_file, delimiter=',').mean()
        aad_loda_runtime = np.loadtxt(aad_loda_rt_file, delimiter=',').mean()
        aad_if_f1 = np.loadtxt(aad_if_f1_file, delimiter=',')
        aad_loda_f1 = np.loadtxt(aad_loda_f1_file, delimiter=',')

        # print everything
        print(f"Dataset: {dataset}")
        print(f"  Our Runtime (Single): {our_runtime_single:.4f}, F1: {our_f1_single:.4f}")
        print(f"  Our Runtime (GMM): {our_runtime_gmm:.4f}, F1: {our_f1_gmm:.4f}")
        print(f"  GAOD Runtime: {gaod_runtime:.4f}, F1: {gaod_f1:.4f}")
        print(f"  AAD IF Runtime: {aad_if_runtime:.4f}, F1: {aad_if_f1.mean():.4f}")
        print(f"  AAD LODA Runtime: {aad_loda_runtime:.4f}, F1: {aad_loda_f1.mean():.4f}")
        print(f"  FIF Runtime: {runtimes_fif:.4f}, F1: {curr_fif_f1:.4f}")

        # --- Plot: One point per method per dataset (outlier-style) ---
        

        methods = ['PMA', 'PMA-MGE', 'GAOD', 'AAD-IF', 'AAD-LODA', 'FIF']
        runtimes = [our_runtime_single, our_runtime_gmm, gaod_runtime, aad_if_runtime, aad_loda_runtime, runtimes_fif]
        f1s = [our_f1_single, our_f1_gmm, gaod_f1, aad_if_f1, aad_loda_f1, curr_fif_f1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'v', 'D', '*']

        for i, method in enumerate(methods):
            xs = runtimes[i]
            ys = f1s[i]
            label = method if method not in plotted_labels else None
            plotted_labels.add(method)
            ax.scatter(xs, ys, color=colors[i], label=label,
                    marker=markers[i], s=marker_size, edgecolor='black', linewidth=0.5, alpha=0.9)
            # ax.annotate(dataset, (xs, ys), fontsize=7, alpha=0.5, textcoords="offset points", xytext=(0, 4), ha='center')

    ax.set_xlabel("Runtime (seconds)", fontsize=ref_font_size)
    ax.set_ylabel("AUPRC Score", fontsize=ref_font_size)
    if dataset == 'cifar':
        ax.set_title("AUPRC vs Runtime - {}_ship".format(dataset), fontsize=ref_font_size+4)
    elif dataset == 'fashion':
        ax.set_title("AUPRC vs Runtime - {}_bag".format(dataset), fontsize=ref_font_size+4)
    else:
        ax.set_title("AUPRC vs Runtime - FMNIST Rest-vs-One".format(dataset), fontsize=ref_font_size+4)
    #ax.set_xscale("log")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    # ax.legend(loc='upper right', frameon=True, fontsize=ref_font_size-2)

            
    os.makedirs('rt_plots', exist_ok=True)
    plt.ylim(0.0, 1.05)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('rt_plots/fashion_aucpr_runtime_grouped_ovr.png'.format(dataset), format='png', dpi=300)
    plt.show()
    plt.close()


        








if __name__ == '__main__':
    #main_plot_diff_k()
    # save_suffixes = ['normed_nok_full_deweight', 'normed_nok_k3_deweight', 'normed_nok_k5_deweight', 'normed_nok_k10_deweight', 'normed_nok_k15_deweight', 'normed_nok_k20_deweight']
    # for suffix in save_suffixes:
    #     main(save_suffix=suffix)
    # main(save_suffix='normed_fullk_99_deweight')
    # main(save_suffix='normed_fullk_95_deweight')
    main_plot_runtimes_vs_f1()
    main_plot_runtimes_vs_prauc()
