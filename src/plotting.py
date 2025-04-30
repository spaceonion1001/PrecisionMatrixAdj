import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import auc
sns.set()

base_dir_aad = '/mnt/c/Users/xdink/Documents/Research/ad_examples-master/ad_examples-master/temp/aad/'
base_dir_boost = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/figures/debugging_figs_true/'
save_dir = '/mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/figures/aad_plots/'

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


if __name__ == '__main__':
    #main_plot_diff_k()
    # save_suffixes = ['normed_nok_full_deweight', 'normed_nok_k3_deweight', 'normed_nok_k5_deweight', 'normed_nok_k10_deweight', 'normed_nok_k15_deweight', 'normed_nok_k20_deweight']
    # for suffix in save_suffixes:
    #     main(save_suffix=suffix)
    main(save_suffix='normed_fullk_99_deweight')
    main(save_suffix='normed_fullk_95_deweight')