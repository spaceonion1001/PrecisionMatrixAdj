import numpy as np
import scipy.stats as stats

def sim_gmm(N=10000, k=2, dim=80, seed=42):
    np.random.seed(42)
    data = []
    labels = []
    while len(data) < N:
        label = np.random.choice(np.arange(k))
        if label == 0:
            curr_point = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=1)
        elif label == 1:
            curr_point = np.random.multivariate_normal(mean=np.ones(dim)*100, cov=np.eye(dim), size=1)
        data.append(curr_point)
        labels.append(label)
    
    return np.array(data).squeeze(), np.array(labels)
