#!/bin/bash

# Run AAD on the dataset

##################### PMA #####################
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data imagenet --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data unsw --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data campaign --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data bank --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data nslkdd --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data msl --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_airplane --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_bird --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_boot --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_sandal --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data mnist --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight


##################### PMA-MGE #####################
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data imagenet --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data unsw --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data campaign --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data bank --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data nslkdd --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data msl --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_airplane --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_bird --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_boot --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_sandal --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data mnist --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight


##################### PMA CIFAR + Fashion #####################
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_0 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_1 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_2 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_3 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_4 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_5 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_6 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_7 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_8 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_9 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight

python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_0 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_1 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_2 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_3 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_4 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_5 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_6 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_7 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_8 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_9 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight

##################### PMA-MGE CIFAR + Fashion #####################
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_0 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_1 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_2 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_3 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_4 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_5 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_6 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_7 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_8 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data cifar_9 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight

python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_0 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_1 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_2 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_3 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_4 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_5 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_6 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_7 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_8 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight
python src/aad_metric_gmm.py --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data --data fashion_9 --iters 50 --k 3 --v 1e-7 --original_mu --nom_deweight --conserv_thresh 50 --true_feedback --full_k --v2 --save_suffix v2_nominal_deweight


##################### PMA SOEL CIFAR #####################
for i in {0..9}; do
  for j in {40..49}; do
    python src/aad_metric.py \
      --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data \
      --data cifar_dlpaper_${i}_${j} \
      --iters 50 \
      --k 3 \
      --v 1e-7 \
      --original_mu \
      --nom_deweight \
      --conserv_thresh 50 \
      --true_feedback \
      --full_k \
      --v2 \
      --save_suffix v2_nominal_deweight
  done
done

##################### PMA-MGE SOEL CIFAR #####################
for i in {0..9}; do
  for j in {40..49}; do
    python src/aad_metric_gmm.py \
      --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data \
      --data cifar_dlpaper_${i}_${j} \
      --iters 50 \
      --k 3 \
      --v 1e-7 \
      --original_mu \
      --nom_deweight \
      --conserv_thresh 50 \
      --true_feedback \
      --full_k \
      --v2 \
      --save_suffix v2_nominal_deweight
  done
done

##################### PMA SOEL Fashion #####################
for i in {0..9}; do
  for j in {40..49}; do
    python src/aad_metric.py \
      --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data \
      --data fashion_dlpaper_${i}_${j} \
      --iters 50 \
      --k 3 \
      --v 1e-7 \
      --original_mu \
      --nom_deweight \
      --conserv_thresh 50 \
      --true_feedback \
      --full_k \
      --v2 \
      --save_suffix v2_nominal_deweight
  done
done

##################### PMA-MGE SOEL Fashion #####################
for i in {0..9}; do
  for j in {40..49}; do
    python src/aad_metric_gmm.py \
      --data_path /mnt/c/Users/xdink/Documents/Research/OnlineMetricLearning/data \
      --data fashion_dlpaper_${i}_${j} \
      --iters 50 \
      --k 3 \
      --v 1e-7 \
      --original_mu \
      --nom_deweight \
      --conserv_thresh 50 \
      --true_feedback \
      --full_k \
      --v2 \
      --save_suffix v2_nominal_deweight
  done
done

