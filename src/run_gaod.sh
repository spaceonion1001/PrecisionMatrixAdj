#!/bin/bash

data_path="/path/to/data"

################################
# rvo, ood, and tabular section
python src/graph_ad.py --data_path "$data_path" --data bank
python src/graph_ad.py --data_path "$data_path" --data campaign
python src/graph_ad.py --data_path "$data_path" --data cifar
python src/graph_ad.py --data_path "$data_path" --data fashion
python src/graph_ad.py --data_path "$data_path" --data nslkdd
python src/graph_ad.py --data_path "$data_path" --data unsw
python src/graph_ad.py --data_path "$data_path" --data imagenet
python src/graph_ad.py --data_path "$data_path" --data msl
python src/graph_ad.py --data_path "$data_path" --data cifar_airplane
python src/graph_ad.py --data_path "$data_path" --data cifar_bird
python src/graph_ad.py --data_path "$data_path" --data fashion_boot
python src/graph_ad.py --data_path "$data_path" --data fashion_sandal
python src/graph_ad.py --data_path "$data_path" --data mnist

################################
# ovr cifar and fashion section
python src/graph_ad.py --data_path "$data_path" --data cifar_0
python src/graph_ad.py --data_path "$data_path" --data cifar_1
python src/graph_ad.py --data_path "$data_path" --data cifar_2
python src/graph_ad.py --data_path "$data_path" --data cifar_3
python src/graph_ad.py --data_path "$data_path" --data cifar_4
python src/graph_ad.py --data_path "$data_path" --data cifar_5
python src/graph_ad.py --data_path "$data_path" --data cifar_6
python src/graph_ad.py --data_path "$data_path" --data cifar_7
python src/graph_ad.py --data_path "$data_path" --data cifar_8
python src/graph_ad.py --data_path "$data_path" --data cifar_9

python src/graph_ad.py --data_path "$data_path" --data fashion_0
python src/graph_ad.py --data_path "$data_path" --data fashion_1
python src/graph_ad.py --data_path "$data_path" --data fashion_2
python src/graph_ad.py --data_path "$data_path" --data fashion_3
python src/graph_ad.py --data_path "$data_path" --data fashion_4
python src/graph_ad.py --data_path "$data_path" --data fashion_5
python src/graph_ad.py --data_path "$data_path" --data fashion_6
python src/graph_ad.py --data_path "$data_path" --data fashion_7
python src/graph_ad.py --data_path "$data_path" --data fashion_8
python src/graph_ad.py --data_path "$data_path" --data fashion_9
