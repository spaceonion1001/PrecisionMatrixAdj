import numpy as np 
import pandas as pd
import os

from numba import njit
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, StandardScaler
np.random.seed(42)

@njit
def mahalanobis_fast_uv(u, v, inv_cov):
    """
    Compute the Mahalanobis distance between two points u and v.

    Parameters:
    - u: numpy array of shape (n_features,)
    - v: numpy array of shape (n_features,)
    - inv_cov: precomputed inverse covariance matrix of shape (n_features, n_features)

    Returns:
    - Mahalanobis distance (scalar)
    """
    delta = u - v  # Compute difference vector
    dist = np.sqrt(np.dot(np.dot(delta, inv_cov), delta.T))  # Efficient quadratic form calculation
    return dist

def mahalanobis_fast_batch(X, center, inv_cov):
    diffs = X - center
    mahal_dists = np.einsum('ij,jk->i', diffs, np.dot(inv_cov, diffs.T))
    return np.sqrt(mahal_dists)

def l2_fast_batch(X, center):
    return np.linalg.norm(X - center, axis=1)


@njit
def low_rank_correction(precision_matrix, top_features, eta=0.1):
    """
    Applies a low-rank correction to the precision matrix to downweight selected features.

    Parameters:
    - precision_matrix: (d, d) array, the inverse covariance matrix (Σ⁻¹)
    - top_features: list of indices of features to deweight
    - eta: float, scaling factor for correction (should be small to preserve PSD)

    Returns:
    - corrected_precision: (d, d) array, modified precision matrix
    """
    print("Low Rank Correction")
    d = precision_matrix.shape[0]
    
    # Construct the V matrix: each selected feature contributes a unit vector
    V = np.zeros((d, len(top_features)))
    for i, feature in enumerate(top_features):
        V[feature, i] = 1  # Unit vector along the chosen feature

    # Compute the low-rank correction
    correction = eta * (V @ V.T)
    
    # Ensure PSD by projecting onto the PSD cone if needed
    corrected_precision = precision_matrix - correction

    # Check if PSD, and project if necessary
    eigvals, eigvecs = np.linalg.eigh(corrected_precision)
    eigvals = np.maximum(eigvals, 1e-6)  # Ensure all eigenvalues are non-negative
    corrected_precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return corrected_precision

@njit
def eigenvalue_softening(precision_matrix, top_features, alpha=0.1):
    """
    Softens eigenvalues of the precision matrix for selected features.

    Parameters:
    - precision_matrix: (d, d) array, the inverse covariance matrix (Σ⁻¹)
    - top_features: list of indices of features to deweight
    - alpha: float, fraction by which to reduce eigenvalues (0 < alpha < 1)

    Returns:
    - softened_precision: (d, d) array, modified precision matrix
    """
    print("Eigenvalue Softening")
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(precision_matrix)
    
    # Reduce eigenvalues corresponding to top-contributing features
    for feature in top_features:
        eigvals[feature] *= (1 - alpha)  # Reduce eigenvalue by alpha fraction

    # Ensure PSD (all eigenvalues must be non-negative)
    eigvals = np.maximum(eigvals, 1e-6)

    # Reconstruct the precision matrix
    softened_precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return softened_precision

@njit
def hybrid_precision_update(precision_matrix, top_features, eta=0.5, alpha=0.3):
    """ Combines low-rank correction and eigenvalue softening for more aggressive deweighting """
    d = precision_matrix.shape[0]

    # Apply Low-Rank Correction
    V = np.zeros((d, len(top_features)))
    for i, feature in enumerate(top_features):
        V[feature, i] = 1  # Unit vectors

    correction = eta * (V @ V.T)
    corrected_precision = precision_matrix - correction

    # Ensure PSD after low-rank correction
    eigvals, eigvecs = np.linalg.eigh(corrected_precision)
    eigvals = np.maximum(eigvals, 1e-6)  # Ensure PSD
    corrected_precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Apply Eigenvalue Softening
    eigvals[top_features] *= (1 - alpha)
    eigvals = np.maximum(eigvals, 1e-6)  # Ensure PSD again
    softened_precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return softened_precision

@njit
def w_precision_update(precision_matrix, top_features, direction_vector):
    # top features are the indices of the top features
    # zero out the non top features in direction vector
    d = direction_vector.shape[0]
    for i in range(d):
        if i not in top_features:
            direction_vector[i] = 0.0
    # normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    direction_vector = np.expand_dims(direction_vector, 1)
    Z = direction_vector @ direction_vector.T
    # make sure Z is rank one trace one
    #lamb_min_a = np.min(np.linalg.eigvals(precision_matrix))
    lamb_min_a = np.min(np.linalg.eigh(precision_matrix)[0])
    #lamb_max_b = np.max(np.linalg.eigvals(Z))
    lamb_max_b = 1.0

    # #############
    # # COMMENT ME OUT FOR NORMAL BEHAVIOR
    # # OPTIONAL EIGENVALUE SCALING
    # eigenvalues, _ = np.linalg.eigh(precision_matrix)
    # # Normalize the eigenvalues to create a scaling vector w
    # eigenvalue_scaling = eigenvalues / np.sum(eigenvalues)  # Normalize eigenvalues
    # # Create a scaling vector `w_vector` that scales based on the eigenvalues
    # w_vector = eigenvalue_scaling * (lamb_min_a / lamb_max_b) - 1e-5
    # W_diag = np.diag(w_vector)
    # adjusted_precision = precision_matrix - W_diag @ Z @ W_diag
    # ##############

    ############
    # UNCOMMENT ME FOR NORMAL BEHAVIOR
    w = (lamb_min_a / lamb_max_b) - 1e-5 # w < lamb_min_a / lamb_max_b

    adjusted_precision = precision_matrix - w * Z
    ##########

    return adjusted_precision

@njit
def is_pos_def(A):
    if is_symmetric(A):
        try:
            np.linalg.cholesky(A)
            return True
        #except np.linalg.LinAlgError:
        except:
            return False
    else:
        return False
    
@njit
def is_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

#@njit
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval <= 0.0] = 1e-8

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def load_census_data(args):
    path = args.data_path

    census_path = os.path.join(path, 'census_1.csv')
    data = pd.read_csv(census_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_wine_data(args):
    path = args.data_path

    wine_path = os.path.join(path, 'wine.data')
    data = pd.read_csv(wine_path, header=None)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    ss = StandardScaler()
    features = ss.fit_transform(features)

    return data.values, features.astype(float), classes

def load_unsw_data(args):
    print("*** LOADING UNSW DATA ***")
    path = args.data_path

    unsw_path = os.path.join(path, 'unsw_1.csv')
    data = pd.read_csv(unsw_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))
    

    return data.values, features.astype(float), classes

def load_kddcup_data(args):
    path = args.data_path

    kddcup_path = os.path.join(path, 'kddcup_1.csv')
    data = pd.read_csv(kddcup_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_campaign_data(args):
    path = args.data_path

    campaign_path = os.path.join(path, 'campaign_1.csv')
    data = pd.read_csv(campaign_path)
    features = data.values[:, 1:]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))

    return data.values, features.astype(float), classes

def load_kdd_donors_data(args):
    path = args.data_path

    kdd_path = os.path.join(path, 'KDD2014_donors_10feat_nomissing_normalised.csv')
    data = pd.read_csv(kdd_path)
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes


def load_bank_data(args):
    path = args.data_path

    bank_path = os.path.join(path, 'bank-additional-full_normalised.csv')
    data = pd.read_csv(bank_path)
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes

def load_retinopathy_data(args):
    path = args.data_path

    retinopathy_path = os.path.join(path, 'retinopathy.arff')
    data = arff.loadarff(retinopathy_path)
    data = pd.DataFrame(data[0]).iloc[:, 1:]
    data['Class'] = data['Class'].str.decode('utf-8').astype('int')
    
    features = data.values[:, :-1]
    classes = data.values[:, -1]

    return data.values, features, classes

def load_cancer_data(args):
    path = args.data_path

    cancer_path = os.path.join(path, 'breast_cancer.DATA')
    data = pd.read_csv(cancer_path, header=None).iloc[:, 1:]
    features = data.values[:, 1:].astype(float)
    le = LabelEncoder()
    classes = np.array(le.fit_transform(data.values[:, 0]))
    classes = 1 - classes

    return data.values, features, classes

def load_yeast_data(args):
    path = args.data_path

    yeast_path = os.path.join(path, 'yeast_1.csv')
    data = pd.read_csv(yeast_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_abalone_data(args):
    path = args.data_path

    abalone_path = os.path.join(path, 'abalone_1.csv')
    data = pd.read_csv(abalone_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_thyroid_data(args):
    path = args.data_path

    thyroid_path = os.path.join(path, 'ann_thyroid_1v3_1.csv')
    data = pd.read_csv(thyroid_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cardio_data(args):
    path = args.data_path

    cardio_path = os.path.join(path, 'cardiotocography_1_1.csv')
    data = pd.read_csv(cardio_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_mammography_data(args):
    path = args.data_path

    mammography_path = os.path.join(path, 'mammography_1.csv')
    data = pd.read_csv(mammography_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_weather_data(args):
    path = args.data_path

    weather_path = os.path.join(path, 'weather_1.csv')
    data = pd.read_csv(weather_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cifar_data(args):
    path = args.data_path

    cifar_path = os.path.join(path, 'cifar_1.csv')
    data = pd.read_csv(cifar_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cifar_numbered_data(args, num=0):
    path = args.data_path + '/vision_processed/'

    cifar_path = os.path.join(path, 'cifar_{}_1.csv'.format(num))
    data = pd.read_csv(cifar_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cifar_numbered_data_dlpaper(args, cnum=0, itr=0):
    path = args.data_path + '/vision_processed/embeddings/'

    cifar_path = os.path.join(path, 'cifar10_feat_{}_{}_train.npz'.format(cnum, itr))
    cifar_path_test = os.path.join(path, 'cifar10_feat_{}_{}_test.npz'.format(cnum, itr))
    data = np.load(cifar_path)
    data_test = np.load(cifar_path_test)
    features = data['embs']
    labels = data['labels']
    features_test = data_test['embs']
    labels_test = data_test['labels']

    return features.astype(float), labels.astype(int), features_test.astype(float), labels_test.astype(int)

def load_cifar_numbered_data_dlpaper_seed(args, cnum=0, seed=0):
    path = args.data_path + '/vision_processed/defaults/'

    cifar_path = os.path.join(path, 'cifar10_{}_{}_train.npz'.format(cnum, seed))
    cifar_path_test = os.path.join(path, 'cifar10_{}_{}_test.npz'.format(cnum, seed))
    data = np.load(cifar_path)
    data_test = np.load(cifar_path_test)
    features = data['data']
    labels = data['labels']
    features_test = data_test['data']
    labels_test = data_test['labels']

    return features.astype(float), labels.astype(int), features_test.astype(float), labels_test.astype(int)

def load_fashion_numbered_data_dlpaper_seed(args, cnum=0, seed=0):
    path = args.data_path + '/vision_processed/defaults/'

    cifar_path = os.path.join(path, 'fashion_{}_{}_train.npz'.format(cnum, seed))
    cifar_path_test = os.path.join(path, 'fashion_{}_{}_test.npz'.format(cnum, seed))
    data = np.load(cifar_path)
    data_test = np.load(cifar_path_test)
    features = data['data']
    labels = data['labels']
    features_test = data_test['data']
    labels_test = data_test['labels']

    return features.astype(float), labels.astype(int), features_test.astype(float), labels_test.astype(int)

def load_fashion_numbered_data(args, num=0):
    path = args.data_path + '/vision_processed/'

    fashion_path = os.path.join(path, 'fashion_{}_1.csv'.format(num))
    data = pd.read_csv(fashion_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cifar_airplane_data(args):
    path = args.data_path

    cifar_path = os.path.join(path, 'cifar_1_airplane.csv')
    data = pd.read_csv(cifar_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cifar_bird_data(args):
    path = args.data_path

    cifar_path = os.path.join(path, 'cifar_1_bird.csv')
    data = pd.read_csv(cifar_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_fashion_data(args):
    path = args.data_path

    fashion_path = os.path.join(path, 'fashion_1.csv')
    data = pd.read_csv(fashion_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_fashion_boot_data(args):
    path = args.data_path

    fashion_path = os.path.join(path, 'fashion_1_boot.csv')
    data = pd.read_csv(fashion_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_fashion_sandal_data(args):
    path = args.data_path

    fashion_path = os.path.join(path, 'fashion_1_sandal.csv')
    data = pd.read_csv(fashion_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_mnist_data(args):
    path = args.data_path

    mnist_path = os.path.join(path, 'mnist_1.csv')
    data = pd.read_csv(mnist_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_svhn_data(args):
    path = args.data_path

    svhn_path = os.path.join(path, 'svhn_1.csv')
    data = pd.read_csv(svhn_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_cover_data(args):
    path = args.data_path

    cover_path = os.path.join(path, 'covtype_1.csv')
    data = pd.read_csv(cover_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_higgs_data(args):
    path = args.data_path

    higgs_path = os.path.join(path, 'higgs_1.csv')
    data = pd.read_csv(higgs_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_spam_data(args):
    path = args.data_path

    spam_path = os.path.join(path, 'spam_1.csv')
    data = pd.read_csv(spam_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_oxford_data(args):
    path = args.data_path

    oxford_path = os.path.join(path, 'oxford_1.csv')
    data = pd.read_csv(oxford_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_imagenet_data(args):
    path = args.data_path

    in_path = os.path.join(path, 'imagenet_1.csv')
    data = pd.read_csv(in_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_msl_data(args):
    path = args.data_path

    in_path = os.path.join(path, 'msl_1.csv')
    data = pd.read_csv(in_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes

def load_nslkdd_data(args):
    path = args.data_path

    nslkdd_path = os.path.join(path, 'nslkdd_1.csv')
    data = pd.read_csv(nslkdd_path)
    features = data.values[:, 1:]
    labels = data.values[:, 0]
    le = LabelEncoder()
    classes = np.array(le.fit_transform(labels))

    return data.values, features.astype(float), classes


def add_epsilon_noise(features):
    print("Adding Epsilon Noise {} ...".format(features.shape))
    for j in range(features.shape[1]):
        features[:, j] += np.random.normal(scale=1e-6, size=features.shape[0])
    
    return features

def remove_bad_features(features):
    print("Removing degenerate features {}".format(features.shape))
    stds = np.std(features, axis=0)
    features = features[:, stds != 0.0]
    print("New shape {}".format(features.shape))
    
    return features

# def smart_sampling(features, labels, num_anoms=10, num_nominals=100):
#     print("Sampling data - margin based (Manhattan Distance)")

#     # calculate Manhattan distance between all points and the mean
#     feat_mean = np.mean(features, axis=0)
#     dist_vec = np.zeros((features.shape[0],))

#     for i in range(features.shape[0]):
#         curr_dist = np.sum(np.abs(features[i] - feat_mean))
#         dist_vec[i] = curr_dist
    
#     # get indices of labels with anomaly class (designated class 0)
#     anom_indices = np.asarray(labels == 0).nonzero()[0]
#     nominal_indices = np.asarray(labels != 0).nonzero()[0]


#     # need to get top K or whatever indices of nominals close to mean and far from mean
#     # get anomalies close to mean
#     # build triplets of these
#     dist_vec_anoms = dist_vec[anom_indices]
#     dist_vec_nominals = dist_vec[nominal_indices]
#     features_anoms = features[anom_indices, :]
#     features_nominals = features[nominal_indices, :]

#     anoms_sorted = np.argsort(dist_vec_anoms)[::-1]
#     nominals_sorted = np.argsort(dist_vec_nominals)[::-1]

#     keep_anomalies_idx = anoms_sorted[-num_anoms:]
#     keep_nominals_far_idx = nominals_sorted[0:num_nominals]
#     keep_nominals_close_idx = nominals_sorted[-num_nominals:]
#     keep_nominals_total_idx = np.array(list(set(np.append(keep_nominals_far_idx, keep_nominals_close_idx))))
    
#     anomaly_pts = features_anoms[keep_anomalies_idx, :]
#     nominal_pts = features_nominals[keep_nominals_total_idx, :]
#     anom_labels = np.zeros(anomaly_pts.shape[0])
#     nom_labels = np.ones(nominal_pts.shape[0])
#     total_feats = np.concatenate((anomaly_pts, nominal_pts), 0)
#     total_labels = np.concatenate((anom_labels, nom_labels), 0)

#     return total_feats, total_labels

    
