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

base_dir_results = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_dl_transduct/'
base_dir_runtime = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_runtimes_dl_transduct/'
base_dir_results_gmm = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_gmms_dl_transduct/'
base_dir_runtime_gmm = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/results_runtimes_dl_gmms_transduct/'
base_dir_dl = '/mnt/c/Users/xdink/Documents/Research/Active-SOEL/NTL/results/'
base_dir_dl_runtime = '/mnt/c/Users/xdink/Documents/Research/Active-SOEL/NTL/runtimes_single_query/'
base_dir_dl_runtime_total = '/mnt/c/Users/xdink/Documents/Research/Active-SOEL/NTL/runtimes_backup/'



def process_cifar_fashion_results_fancy():
    ref_font_size = 22
    tick_label_size = 18
    cifar_datasets = ['CIFAR_dlpaper_{}'.format(x) for x in range(10)]
    # fashion_datasets = ['fashion_dlpaper_{}'.format(x) for x in range(10)]
    datasets = cifar_datasets
    # datasets = fashion_datasets

    f1_rows = []
    auc_pr_rows = []
    dl_f1_all = []
    our_f1_all = []
    our_f1_gmm_all = []
    dl_ap_all = []
    our_ap_all = []
    our_ap_gmm_all = []

    seeds = np.arange(0, 10)
    for i, dataset in enumerate(datasets):
        dataset_lower = dataset.lower()
        our_f1s = []
        our_f1s_gmm = []
        our_aps = []
        our_aps_gmm = []
        for seed in seeds:
            our_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_{40+seed}_v2_nominal_deweight.txt')
            gmm_f1_file = os.path.join(base_dir_results_gmm, f'precision_scores_{dataset_lower}_{40+seed}_v2_nominal_deweight.txt')

            our_f1_single = get_last_seed_f1(our_f1_file)
            our_f1_gmm = get_last_seed_f1(gmm_f1_file)
            our_ap_single = get_last_seed_ap(our_f1_file)
            our_ap_gmm = get_last_seed_ap(gmm_f1_file)

            our_f1s.append(our_f1_single)
            our_f1s_gmm.append(our_f1_gmm)
            our_aps.append(our_ap_single)
            our_aps_gmm.append(our_ap_gmm)

        dl_f1_file = os.path.join(base_dir_dl, f'cifar10_feat_{i}f1.txt')
        dl_ap_file = os.path.join(base_dir_dl, f'cifar10_feat_{i}avgprec.txt')
        # dl_f1_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}f1.txt')
        # dl_ap_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}avgprec.txt')
        dl_f1 = np.loadtxt(dl_f1_file).mean()
        dl_ap = np.loadtxt(dl_ap_file).mean()

        our_f1_mean = np.mean(our_f1s)
        our_f1_gmm_mean = np.mean(our_f1s_gmm)
        our_ap_mean = np.mean(our_aps)
        our_ap_gmm_mean = np.mean(our_aps_gmm)

        dl_f1_all.append(dl_f1)
        our_f1_all.append(our_f1_mean)
        our_f1_gmm_all.append(our_f1_gmm_mean)
        dl_ap_all.append(dl_ap)
        our_ap_all.append(our_ap_mean)
        our_ap_gmm_all.append(our_ap_gmm_mean)

    # X-axis: class labels 0-9
    x = np.arange(10)

    # Bar width and positioning
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting
    ax.bar(x - width, dl_f1_all, width, label='SOEL', color='#e377c2')
    ax.bar(x, our_f1_all, width, label='PMA', color='#1f77b4')
    ax.bar(x + width, our_f1_gmm_all, width, label='PMA-MGE', color='#ff7f0e')

    # X-axis labels and ticks
    ax.set_xlabel('CIFAR-10 Class Index', fontsize=ref_font_size)
    ax.set_ylabel('F1 Score', fontsize=ref_font_size)
    ax.set_title('F1 Scores vs SOEL - CIFAR-10', fontsize=ref_font_size + 4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Legend
    # ax.legend(frameon=True, loc='upper right', fontsize=ref_font_size - 2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )

    # Save and show
    plt.tight_layout()
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('f1_barplot_cifar.png', dpi=300)
    plt.show()

    # Write AUCPR table
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting
    ax.bar(x - width, dl_ap_all, width, label='SOEL', color='#e377c2')
    ax.bar(x, our_ap_all, width, label='PMA', color='#1f77b4')
    ax.bar(x + width, our_ap_gmm_all, width, label='PMA-MGE', color='#ff7f0e')

    # X-axis labels and ticks
    ax.set_xlabel('CIFAR-10 Class Index', fontsize=ref_font_size)
    ax.set_ylabel('AUPRC Score', fontsize=ref_font_size)
    ax.set_title('AUPRC Scores vs SOEL - CIFAR-10', fontsize=ref_font_size + 4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Legend
    # ax.legend(frameon=True, loc='upper right', fontsize=ref_font_size - 2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )

    # Save and show
    plt.tight_layout()
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('aucpr_barplot_cifar.png', dpi=300)
    plt.show()
    plt.close()










    # cifar_datasets = ['CIFAR_dlpaper_{}'.format(x) for x in range(10)]
    fashion_datasets = ['fashion_dlpaper_{}'.format(x) for x in range(10)]
    # datasets = cifar_datasets
    datasets = fashion_datasets

    f1_rows = []
    auc_pr_rows = []
    dl_f1_all = []
    our_f1_all = []
    our_f1_gmm_all = []
    dl_ap_all = []
    our_ap_all = []
    our_ap_gmm_all = []

    seeds = np.arange(0, 10)
    for i, dataset in enumerate(datasets):
        dataset_lower = dataset.lower()
        our_f1s = []
        our_f1s_gmm = []
        our_aps = []
        our_aps_gmm = []
        for seed in seeds:
            our_f1_file = os.path.join(base_dir_results, f'precision_scores_{dataset_lower}_{40+seed}_v2_nominal_deweight.txt')
            gmm_f1_file = os.path.join(base_dir_results_gmm, f'precision_scores_{dataset_lower}_{40+seed}_v2_nominal_deweight.txt')

            our_f1_single = get_last_seed_f1(our_f1_file)
            our_f1_gmm = get_last_seed_f1(gmm_f1_file)
            our_ap_single = get_last_seed_ap(our_f1_file)
            our_ap_gmm = get_last_seed_ap(gmm_f1_file)

            our_f1s.append(our_f1_single)
            our_f1s_gmm.append(our_f1_gmm)
            our_aps.append(our_ap_single)
            our_aps_gmm.append(our_ap_gmm)

        dl_f1_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}f1.txt')
        dl_ap_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}avgprec.txt')
        # dl_f1_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}f1.txt')
        # dl_ap_file = os.path.join(base_dir_dl, f'fmnist_feat_{i}avgprec.txt')
        dl_f1 = np.loadtxt(dl_f1_file).mean()
        dl_ap = np.loadtxt(dl_ap_file).mean()

        our_f1_mean = np.mean(our_f1s)
        our_f1_gmm_mean = np.mean(our_f1s_gmm)
        our_ap_mean = np.mean(our_aps)
        our_ap_gmm_mean = np.mean(our_aps_gmm)

        dl_f1_all.append(dl_f1)
        our_f1_all.append(our_f1_mean)
        our_f1_gmm_all.append(our_f1_gmm_mean)
        dl_ap_all.append(dl_ap)
        our_ap_all.append(our_ap_mean)
        our_ap_gmm_all.append(our_ap_gmm_mean)

    # X-axis: class labels 0-9
    x = np.arange(10)

    # Bar width and positioning
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting
    ax.bar(x - width, dl_f1_all, width, label='SOEL', color='#e377c2')
    ax.bar(x, our_f1_all, width, label='PMA', color='#1f77b4')
    ax.bar(x + width, our_f1_gmm_all, width, label='PMA-MGE', color='#ff7f0e')

    # X-axis labels and ticks
    ax.set_xlabel('FashionMNIST Class Index', fontsize=ref_font_size)
    ax.set_ylabel('F1 Score', fontsize=ref_font_size)
    ax.set_title('F1 Scores vs SOEL - FashionMNIST', fontsize=ref_font_size + 4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Legend
    # ax.legend(frameon=True, loc='upper right', fontsize=ref_font_size - 2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )

    # Save and show
    plt.tight_layout()
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('f1_barplot_fashion.png', dpi=300)
    plt.show()

    # Write AUCPR table
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting
    ax.bar(x - width, dl_ap_all, width, label='SOEL', color='#e377c2')
    ax.bar(x, our_ap_all, width, label='PMA', color='#1f77b4')
    ax.bar(x + width, our_ap_gmm_all, width, label='PMA-MGE', color='#ff7f0e')

    # X-axis labels and ticks
    ax.set_xlabel('FashionMNIST Class Index', fontsize=ref_font_size)
    ax.set_ylabel('AUPRC Score', fontsize=ref_font_size)
    ax.set_title('AUPRC Scores vs SOEL - FashionMNIST', fontsize=ref_font_size + 4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Legend
    # ax.legend(frameon=True, loc='upper right', fontsize=ref_font_size - 2)
    ax.legend(
        loc='upper left',           # Anchor the legend box to the upper-left corner...
        bbox_to_anchor=(1.02, 1),   # ...but position it just outside the right edge of the plot
        frameon=True,
        fontsize=ref_font_size - 2
    )

    # Save and show
    plt.tight_layout()
    ax.tick_params(axis='both', labelsize=tick_label_size)
    plt.savefig('aucpr_barplot_fashion.png', dpi=300)
    plt.show()
    plt.close()

    # # Write F1 table
    # with open("f1_table_dl_cifar.txt", "w") as f:
    #     f.write("\\begin{tabular}{lccc}\n\\toprule\nDataset & SOEL & PMA & PMA-MGE \\\\\n\\midrule\n")
    #     for i in range(10):
    #         row = [dl_f1_all[i], our_f1_all[i], our_f1_gmm_all[i]]
    #         max_idx = int(np.argmax(row))
    #         row_fmt = [f"{v:.3f}" if j != max_idx else f"\\textbf{{{v:.3f}}}" for j, v in enumerate(row)]
    #         f.write(f"Cifar\\_{i} & {row_fmt[0]} & {row_fmt[1]} & {row_fmt[2]} \\\\\n")
    #     f.write("\\bottomrule\n\\end{tabular}\n")

    # # Write AUPRC table
    # with open("auprc_table_dl_cifar.txt", "w") as f:
    #     f.write("\\begin{tabular}{lccc}\n\\toprule\nDataset & SOEL & PMA & PMA-MGE \\\\\n\\midrule\n")
    #     for i in range(10):
    #         row = [dl_ap_all[i], our_ap_all[i], our_ap_gmm_all[i]]
    #         max_idx = int(np.argmax(row))
    #         row_fmt = [f"{v:.3f}" if j != max_idx else f"\\textbf{{{v:.3f}}}" for j, v in enumerate(row)]
    #         f.write(f"Cifar\\_{i} & {row_fmt[0]} & {row_fmt[1]} & {row_fmt[2]} \\\\\n")
    #     f.write("\\bottomrule\n\\end{tabular}\n")

def average_runtimes_dl(folder):
    cifar_runtimes = []
    fmnist_runtimes = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                try:
                    runtimes = [float(line.strip()) for line in f.readlines()]
                    assert len(runtimes) == 10, f"File {filename} does not have 10 lines"
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
                    continue

                if filename.startswith("cifar"):
                    cifar_runtimes.append(runtimes)
                elif filename.startswith("fmnist"):
                    fmnist_runtimes.append(runtimes)

    # Compute averages
    cifar_avg = np.mean(cifar_runtimes) if cifar_runtimes else None
    fmnist_avg = np.mean(fmnist_runtimes) if fmnist_runtimes else None

    return cifar_avg, fmnist_avg

# def process_cifar_fashion_runtimes():
#     cifar_datasets = ['CIFAR_dlpaper_{}'.format(x) for x in range(10)]
#     fashion_datasets = ['fashion_dlpaper_{}'.format(x) for x in range(10)]
#     datasets = cifar_datasets + fashion_datasets
#     cifar_runtime_dl, fashion_runtime_dl = average_runtimes_dl(base_dir_dl_runtime)
#     seeds = np.arange(40, 50)
#     runtimes = []
#     cifar_runtimes = []
#     cifar_runtimes_gmm = []
#     fashion_runtimes = []
#     fashion_runtimes_gmm = []
#     for dataset in datasets:
#         dataset_lower = dataset.lower()
#         for seed in seeds:
#             our_runtime_file = np.loadtxt(os.path.join(base_dir_runtime, 'runtimes_{}_{}_v2_nominal_deweight.txt'.format(dataset_lower, seed)))
#             our_runtime_file_gmm = np.loadtxt(os.path.join(base_dir_runtime_gmm, 'runtimes_{}_{}_v2_nominal_deweight.txt'.format(dataset_lower, seed)))
#             if 'cifar' in dataset_lower:
#                 cifar_runtimes.append(our_runtime_file)
#                 cifar_runtimes_gmm.append(our_runtime_file_gmm)
#             elif 'fashion' in dataset_lower:
#                 fashion_runtimes.append(our_runtime_file)
#                 fashion_runtimes_gmm.append(our_runtime_file_gmm)
#     cifar_runtime = np.array(cifar_runtimes).mean()
#     cifar_runtime_gmm = np.array(cifar_runtimes_gmm).mean()
#     cifar_runtime = cifar_runtime / 50 # convert to single query runtime
#     cifar_runtime_gmm = cifar_runtime_gmm / 50 # convert to single query runtime
#     fashion_runtime = np.array(fashion_runtimes).mean()
#     fashion_runtime_gmm = np.array(fashion_runtimes_gmm).mean()
#     fashion_runtime = fashion_runtime / 50 # convert to single query runtime
#     fashion_runtime_gmm = fashion_runtime_gmm / 50 # convert to single query runtime

def process_cifar_fashion_runtimes():
    cifar_datasets = ['CIFAR_dlpaper_{}'.format(x) for x in range(10)]
    fashion_datasets = ['fashion_dlpaper_{}'.format(x) for x in range(10)]
    datasets = cifar_datasets + fashion_datasets

    cifar_runtime_dl, fashion_runtime_dl = average_runtimes_dl(base_dir_dl_runtime)
    cifar_runtime_dl_total, fashion_runtime_dl_total = average_runtimes_dl(base_dir_dl_runtime_total)
    seeds = np.arange(40, 50)

    cifar_runtimes = []
    cifar_runtimes_gmm = []
    fashion_runtimes = []
    fashion_runtimes_gmm = []

    for dataset in datasets:
        dataset_lower = dataset.lower()
        for seed in seeds:
            our_runtime_file = np.loadtxt(os.path.join(base_dir_runtime, f'runtimes_{dataset_lower}_{seed}_v2_nominal_deweight.txt'))
            our_runtime_file_gmm = np.loadtxt(os.path.join(base_dir_runtime_gmm, f'runtimes_{dataset_lower}_{seed}_v2_nominal_deweight.txt'))
            if 'cifar' in dataset_lower:
                cifar_runtimes.append(our_runtime_file)
                cifar_runtimes_gmm.append(our_runtime_file_gmm)
            elif 'fashion' in dataset_lower:
                fashion_runtimes.append(our_runtime_file)
                fashion_runtimes_gmm.append(our_runtime_file_gmm)

    cifar_runtime = np.mean(cifar_runtimes) / 50
    cifar_runtime_gmm = np.mean(cifar_runtimes_gmm) / 50
    fashion_runtime = np.mean(fashion_runtimes) / 50
    fashion_runtime_gmm = np.mean(fashion_runtimes_gmm) / 50

    # LaTeX table
    latex_table = r"""\begin{{table}}[h]
    \centering
    \begin{{tabular}}{{lccc}}
    \toprule
    Dataset & SOEL & PMA & PMA-MGE \\
    \midrule
    CIFAR   & {:.3f} & {:.3f} & {:.3f} \\
    Fashion & {:.3f} & {:.3f} & {:.3f} \\
    \bottomrule
    \end{{tabular}}
    \caption{{Average single-query runtime (in seconds) for each algorithm and dataset.}}
    \label{{tab:runtime_comparison}}
    \end{{table}}
    """.format(cifar_runtime_dl, cifar_runtime, cifar_runtime_gmm,
            fashion_runtime_dl, fashion_runtime, fashion_runtime_gmm)
    
    # LaTeX table
    latex_table_total = r"""\begin{{table}}[h]
    \centering
    \begin{{tabular}}{{lccc}}
    \toprule
    Dataset & SOEL & PMA & PMA-MGE \\
    \midrule
    CIFAR   & {:.3f} & {:.3f} & {:.3f} \\
    Fashion & {:.3f} & {:.3f} & {:.3f} \\
    \bottomrule
    \end{{tabular}}
    \caption{{Average total runtime (in seconds) for each algorithm and dataset, budget of 50 queries.}}
    \label{{tab:runtime_comparison}}
    \end{{table}}
    """.format(cifar_runtime_dl_total, cifar_runtime*50, cifar_runtime_gmm*50,
            fashion_runtime_dl_total, fashion_runtime*50, fashion_runtime_gmm*50)


    # Write to file
    with open("runtime_table_soel.txt", "w") as f:
        f.write(latex_table)
    with open("runtime_table_soel_total.txt", "w") as f:
        f.write(latex_table_total)





if __name__ == "__main__":
    process_cifar_fashion_results_fancy()
    # print("Results processed and tables generated.")
    # print(average_runtimes_dl(base_dir_dl_runtime))
    # process_cifar_fashion_runtimes()




        