import numpy as np
from sklearn.utils.extmath import fast_logdet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from numpy.linalg import inv

from numba import njit
from numba_progress import ProgressBar

from utils import *
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from autoencoder import AE, train_model
from torch.utils.data import TensorDataset, DataLoader
import torch

@njit
def my_mahal_squared(u, v, A):
    dist = (u-v).T@A@(u-v)

    return dist

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='C:\\Users\\xdink\\Documents\\Research\\OnlineMetricLearning\\data')
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--burnin_size', type=int, default=100)
    parser.add_argument('--data', type=str, default='census')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--autoencode', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--pca', action='store_true')

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
    else:
        print("Incorrect Dataset...")
        exit(1)

@njit
def true_y(ulabel, vlabel):
    # simple binary function
    if ulabel == vlabel:
        return 0.0
    else:
        return 3.0

def calc_loss(yhat_t, y_t):
    return 0.5*((yhat_t - y_t)**2)

@njit
def calc_ybar(eta, y_t, yhat_t):
    denom = 2*eta*yhat_t

    first = eta*y_t*yhat_t - 1
    inn = (eta*y_t*yhat_t - 1)**2 + 4*eta*(yhat_t)**2
    numer = first + np.sqrt(inn)
    if denom <= 0.0:
        denom = 1e-6

    return numer/denom

@njit
def calc_A_tpo(eta, ybar, y_t, A_t, u_t, v_t):
    z_t = u_t - v_t
    first = eta*(ybar - y_t)
    second = A_t@z_t@z_t.T@A_t
    numer = first*second

    dsecond = z_t.T@A_t@z_t
    denom = 1 + first*dsecond

    A_tpo = A_t - numer/denom
    if not is_pos_def(A_tpo):
        A_tpo = get_near_psd(A_tpo)
    
    assert is_pos_def(A_tpo)
    assert is_symmetric(A_tpo)

    return A_tpo

def calc_mahal_avg(nominal_test, outlier_test, A_t):
    total_summand = 0.0
    denom = 0
    running_avg = 0.0
    dist_arr = []
    for u in tqdm(nominal_test):
        u = np.expand_dims(u[:-1], 1)
        for v in outlier_test:
            v = np.expand_dims(v[:-1], 1)
            curr_dist = np.sqrt(my_mahal_squared(u, v, A_t)[0][0])
            dist_arr.append(curr_dist)
            total_summand += curr_dist
            denom += 1
            running_avg = total_summand/denom
    dist_arr = np.array(dist_arr)

    print("Max {} Min {} Median {} Mean {}".format(dist_arr.max(), dist_arr.min(), np.median(dist_arr), dist_arr.mean()))
    return dist_arr

@njit(nogil=True)
def run_iterations_set(eta, first_data, second_data, A_t, progress_proxy, num_iterations):
    for i in range(num_iterations):
        u = first_data[i, :]
        label_u = 0
        u = np.expand_dims(u[:-1], 1)
        progress_proxy.update(1)
        for j in range(second_data.shape[0]):
            v = second_data[i, :]
            label_v = 0
            y_t = true_y(label_u, label_v)
            v = np.expand_dims(v[:-1], 1)
            #yhat_t = mahalanobis(u, v, A_t)
            yhat_t = np.sqrt(my_mahal_squared(u, v, A_t)[0][0])
            ybar = calc_ybar(eta=eta, y_t=y_t, yhat_t=yhat_t)
            A_tpo = calc_A_tpo(eta=eta, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
            A_t = A_tpo
    return A_t

def init_covar(data):
    A_t = np.cov(data.T)
    A_t = A_t + 0.05*np.eye(A_t.shape[0])

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

def get_queries(anomaly_scores, features, labels, k=5):
    ind = np.argpartition(anomaly_scores, -k)[-k:]
    topk_scores = anomaly_scores[ind]
    topk_feats = features[ind]
    topk_labels = labels[ind]

    return ind, topk_feats, topk_labels

def query_set_update(features, labels, A_t, eta):
    print("Query Set Size {}".format(features.shape[0]))
    for i in range(features.shape[0]-1):
        u = np.expand_dims(features[i, :], 1)
        ulabel = labels[i]
        for j in range(i, features.shape[0]):
            v = np.expand_dims(features[j, :], 1)
            vlabel = labels[j]
            y_t = true_y(ulabel, vlabel)
            #yhat_t_my = my_mahal_squared(u, v, A_t).squeeze()
            yhat_t = mahalanobis(u.flatten(), v.flatten(), A_t)
            #yhat_t = np.sqrt(my_mahal_squared(u, v, A_t)[0][0])
            ybar = calc_ybar(eta=eta, y_t=y_t, yhat_t=yhat_t)
            A_tpo = calc_A_tpo(eta=eta, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
            A_t = A_tpo
    
    return A_t




def main(args):
    data, features, labels = resolve_data(args)
    seen_flags = np.zeros_like(labels)
    print(np.unique(labels, return_counts=True)) # 0 is usually anomaly
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    if args.autoencode:
        print(">>> Training Autoencoder on Data... <<<")
        print("Cuda Availability: {}".format(torch.cuda.is_available()))
        model = AE(input_dim=features.shape[1], hidden_dim=args.hidden_size)
        tensor_x = torch.Tensor(features) # transform to torch tensor
        tensor_y = torch.Tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-10)
        model.train()
        model = train_model(model, loss=torch.nn.MSELoss(), optimizer=optimizer, loader=dataloader, epochs=args.epochs)
        model.eval()
        _, reduced_features = model(tensor_x)
        reduced_features = reduced_features.detach().numpy()
        scaler = StandardScaler()
        reduced_features = scaler.fit_transform(reduced_features)
        print("Reduced Feature Shape ", reduced_features.shape)
    
        print(features.shape, reduced_features.shape)
    
        features = reduced_features

    elif args.pca:
        print(">>> Fitting PCA on Data... <<<")
        pca = PCA(n_components=args.hidden_size)
        reduced_features = pca.fit_transform(features)
        scaler = StandardScaler()
        reduced_features = scaler.fit_transform(reduced_features)
        print("Reduced Feature Shape ", reduced_features.shape)
    
        print(features.shape, reduced_features.shape)

        features = reduced_features

    mean_vec = np.mean(features, axis=0)
    A_t = init_covar(features)
    anomaly_scores = calc_anomaly_scores(features=features, A_t=A_t, mean_vec=mean_vec)
    queried_feats = np.empty((0, features.shape[1]))
    queried_labels = np.array([])
    num_queried = 0
    anoms_found = 0
    plotting_tups = []
    plotting_tups.append((0, 0))
    while num_queried < args.budget:
        #print("LOOP START")
        topk_ind, topk_feats, topk_labels = get_queries(anomaly_scores=anomaly_scores, features=features, labels=labels, k=args.k)
        #num_anoms = (topk_labels == 0).sum() # 0 is the label for this data
        #anoms_found += num_anoms
        #plotting_tups.append((num_queried, anoms_found))
        for i in range(topk_feats.shape[0]):
            curr_ind = topk_ind[i]
            curr_feat = topk_feats[i]
            curr_label = topk_labels[i]
            print("Index {} Label {}".format(curr_ind, curr_label))
            queried_feats = np.vstack((queried_feats, np.expand_dims(curr_feat, 0)))
            queried_labels = np.concatenate((queried_labels, np.expand_dims(curr_label, 0)))
            A_t = query_set_update(features=queried_feats, labels=queried_labels, A_t=A_t, eta=args.eta)
            features = np.delete(features, curr_ind, axis=0)
            labels = np.delete(labels, curr_ind)
            anomaly_scores = calc_anomaly_scores(features=features, A_t=A_t, mean_vec=mean_vec)
            num_queried += 1
            topk_ind_int, topk_feats_int, topk_labels_int = get_queries(anomaly_scores=anomaly_scores, features=features, labels=labels, k=args.k)
            #print("Index New {} Label New {}".format(topk_ind_int, topk_labels_int))
            #curr_seen = seen_flags[topk_ind_int]
            num_anoms = (topk_labels_int == 0).sum() # 0 is the label for this data
            anoms_found += num_anoms
            plotting_tups.append((num_queried, anoms_found))
            if num_queried >= args.budget:
                break
        #anomaly_scores = calc_anomaly_scores(features=features, A_t=A_t, mean_vec=mean_vec)
    queries = np.array([x[0] for x in plotting_tups])
    detects = np.array([x[1] for x in plotting_tups])
    if args.autoencode:
        np.savetxt('results/{}/{}'.format(args.data, 'detects_k{}_autoencode.csv'.format(args.k)), detects.astype(int))
    elif args.autoencode:
        np.savetxt('results/{}/{}'.format(args.data, 'detects_k{}_pca.csv'.format(args.k)), detects.astype(int))
    else:
        np.savetxt('results/{}/{}'.format(args.data, 'detects_k{}.csv'.format(args.k)), detects.astype(int))

    

def main_test(args):
    #data, _, _ = load_census_data(args)
    data, _, _ = resolve_data(args)
    print("Data Shape {}".format(data.shape))
    nominal_data = data[np.where(data[:, -1] == 0)]
    outlier_data = data[np.where(data[:, -1] == 1)]
    print("Nominal Data {} Outlier Data {}".format(nominal_data.shape, outlier_data.shape))
    eta = args.eta

    seed = 42
    np.random.seed(42)
    nominal_train, nominal_test = train_test_split(nominal_data, test_size=0.2, random_state=seed)
    outlier_train, outlier_test = train_test_split(outlier_data, test_size=0.2, random_state=seed)
    print("Nom Train {} Nom Test {}".format(nominal_train.shape, nominal_test.shape))
    print("Outlier Train {} Outlier Test {}".format(outlier_train.shape, outlier_test.shape))
    # instantialize distance matrix
    train_set = np.vstack((nominal_train[:, :-1], outlier_train[:, :-1]))
    A_t = np.cov(train_set.T)
    A_t = A_t + 0.05*np.eye(A_t.shape[0])
    #inv_A_t = np.eye(train_set.shape[1])
    # if not is_pos_def(inv_A_t):
    #     inv_A_t = get_near_psd(inv_A_t)
    #A_t = inv(inv_A_t)

    # burn in period
    burn_in_nominal = nominal_data[np.random.choice(np.arange(nominal_test.shape[0]), replace=False, size=args.burnin_size), :]
    burn_in_nominal_second = nominal_data[np.random.choice(np.arange(nominal_test.shape[0]), replace=False, size=args.burnin_size), :]
    print("Performing Burn In")
    with ProgressBar(total=burn_in_nominal.shape[0]) as progress:
        A_t = run_iterations_set(eta=args.eta, first_data=burn_in_nominal, second_data=burn_in_nominal_second, A_t=A_t, progress_proxy=progress, num_iterations=burn_in_nominal.shape[0])
    
    for it in range(10):
        query_set_nominal = nominal_train[np.random.choice(np.arange(nominal_train.shape[0]), replace=False, size=9), :]
        query_set_outlier = outlier_train[np.random.choice(np.arange(outlier_train.shape[0]), replace=False, size=1), :]
        query_set = np.vstack((query_set_nominal, query_set_outlier))
        np.random.shuffle(query_set)
        for u in query_set:
            label_u = int(u[-1])
            u = np.expand_dims(u[:-1], 1)
            for v in query_set:
                label_v = int(v[-1])
                y_t = true_y(label_u, label_v)
                v = np.expand_dims(v[:-1], 1)
                #yhat_t_my = my_mahal_squared(u, v, A_t).squeeze()
                yhat_t = mahalanobis(u, v, A_t)
                #yhat_t = np.sqrt(my_mahal_squared(u, v, A_t)[0][0])
                ybar = calc_ybar(eta=args.eta, y_t=y_t, yhat_t=yhat_t)
                A_tpo = calc_A_tpo(eta=args.eta, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
                A_t = A_tpo

            #print(mahalanobis(u, v, A_t), np.sqrt(my_mahal_squared(u, v, A_t).squeeze()))

    subsample_nom = np.random.choice(np.arange(nominal_test.shape[0]), size=min(1000, nominal_test.shape[0]), replace=False)
    subsample_out = np.random.choice(np.arange(outlier_test.shape[0]), size=min(1000, outlier_test.shape[0]), replace=False)
    dist_arr = calc_mahal_avg(nominal_test=nominal_test[subsample_nom, :], outlier_test=nominal_test[subsample_nom, :], A_t=A_t)
    print("Intra Class Nominal {}".format(dist_arr.mean()))

    dist_arr_outlier = calc_mahal_avg(nominal_test=nominal_test[subsample_nom, :], outlier_test=outlier_test[subsample_out, :], A_t=A_t)
    print("Inter Class {}".format(dist_arr_outlier.mean()))
    fig = plt.figure()
    ax = sns.boxplot(data=[dist_arr, dist_arr_outlier])
    ax.set_xticklabels(['Nominal', 'Anomaly'])
    plt.ylabel('Distance')
    plt.title(args.data+" estimated distances")
    plt.savefig('figures\\boxplot_{}.png'.format(args.data))


            
    
if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)
#     U_set = np.random.rand(10, 10000)
#     V_set = np.random.normal(10, 1, (10, 10000))
#     print(U_set.shape, V_set.shape)
#     #u = np.random.rand(10, 1)
#     #v = np.random.normal(10, 1, (10, 1))
#     inv_A_t = np.cov(np.column_stack((U_set[:, 0:100], V_set[:, 0:100])), bias=False) + 1e-4*np.eye(10)
#     eta = 0.1

#     A_t = inv(inv_A_t)
#     for i in range(U_set.shape[1]):
#         u = np.expand_dims(U_set[:, i], 1)
#         v = np.expand_dims(V_set[:, i], 1)
#         y_t = true_y(0, 1)
#         yhat_t = mahalanobis(u, v, A_t)
#         ybar = calc_ybar(eta=0.05, y_t=y_t, yhat_t=yhat_t)
#         A_tpo = calc_A_tpo(eta=0.05, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
#         A_t = A_tpo
#         #print(mahalanobis(u, v, A_t))

#         j = np.random.choice(np.arange(0, 10000))
#         k  = np.random.choice(np.arange(0, 10000))
#         u = np.expand_dims(U_set[:, j], 1)
#         v = np.expand_dims(U_set[:, k], 1)
#         y_t = true_y(0, 0)
#         yhat_t = mahalanobis(u, v, A_t)
#         ybar = calc_ybar(eta=0.05, y_t=y_t, yhat_t=yhat_t)
#         A_tpo = calc_A_tpo(eta=0.05, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
#         A_t = A_tpo

#         j = np.random.choice(np.arange(0, 10000))
#         k  = np.random.choice(np.arange(0, 10000))
#         u = np.expand_dims(V_set[:, j], 1)
#         v = np.expand_dims(V_set[:, k], 1)
#         y_t = true_y(1, 1)
#         yhat_t = mahalanobis(u, v, A_t)
#         ybar = calc_ybar(eta=0.05, y_t=y_t, yhat_t=yhat_t)
#         A_tpo = calc_A_tpo(eta=0.05, ybar=ybar, y_t=y_t, A_t=A_t, u_t=u, v_t=v)
#         A_t = A_tpo
    
#     j = np.random.choice(np.arange(0, 10000))
#     k = np.random.choice(np.arange(0, 10000))
#     u = np.expand_dims(U_set[:, j], 1)
#     v = np.expand_dims(U_set[:, k], 1)
#     print("Same Class U: {}".format(mahalanobis(u, v, A_t)))

#     u = np.expand_dims(V_set[:, j], 1)
#     v = np.expand_dims(V_set[:, k], 1)
#     print("Same Class V: {}".format(mahalanobis(u, v, A_t)))

#     u = np.expand_dims(U_set[:, j], 1)
#     v = np.expand_dims(V_set[:, k], 1)
#     print("Diff Class U/V: {}".format(mahalanobis(u, v, A_t)))

#     u = np.expand_dims(V_set[:, j], 1)
#     v = np.expand_dims(U_set[:, k], 1)
#     print("Diff Class V/U: {}".format(mahalanobis(u, v, A_t)))

        

    

    