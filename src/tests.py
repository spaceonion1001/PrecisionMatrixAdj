import numpy as np
from scipy.spatial.distance import mahalanobis
from utils import is_pos_def
from statsmodels.stats.moment_helpers import cov2corr
# Function to update the inverse covariance matrix with a new weak learner using an orthogonal vector to u
def update_metric_decrease_distance_orth(S_inv, x1, x2, alpha):
    """
    Update the Mahalanobis metric S^-1 to decrease the distance between x1 and x2 using an orthogonal vector.
    
    Parameters:
    - S_inv: Current inverse covariance matrix (numpy array).
    - x1, x2: Data points (numpy arrays).
    - alpha: Weight for the new weak learner.
    
    Returns:
    - S_inv_new: Updated inverse covariance matrix.
    """
    # Compute the direction vector u
    u = x1 - x2

    # Find an orthogonal vector to u (using the Gram-Schmidt process)
    def find_orthogonal_vector(v, reference):
        """Find an orthogonal vector to the given vector v using the reference vector."""
        v_orthogonal = v - (np.dot(v, reference) / np.dot(reference, reference)) * reference
        return v_orthogonal

    # Generate an initial random vector and make it orthogonal to u
    random_vector = np.random.rand(len(u))
    v_orthogonal = find_orthogonal_vector(random_vector, u)
    print(np.dot(v_orthogonal, u))  # Check that the vectors are orthogonal

    # Normalize the orthogonal vector
    v_orthogonal_norm = np.linalg.norm(v_orthogonal)
    if v_orthogonal_norm == 0:
        raise ValueError("Failed to find an orthogonal vector; try a different random vector.")
    v_orthogonal = v_orthogonal / v_orthogonal_norm

    # Construct the new weak learner (rank-1 trace-1 matrix)
    Z_new = np.outer(v_orthogonal, v_orthogonal)

    # Update the inverse covariance matrix
    S_inv_new = S_inv + alpha * Z_new

    return S_inv_new

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x1, x2, S_inv):
    diff = x1 - x2
    return np.sqrt(diff @ S_inv @ diff)

import numpy as np

# Function to update the inverse covariance matrix with a new weak learner
def update_metric_decrease_distance(S_inv, x1, x2, alpha):
    """
    Update the Mahalanobis metric S^-1 to decrease the distance between x1 and x2.
    
    Parameters:
    - S_inv: Current inverse covariance matrix (numpy array).
    - x1, x2: Data points (numpy arrays).
    - alpha: Weight for the new weak learner.
    
    Returns:
    - S_inv_new: Updated inverse covariance matrix.
    """
    # Compute the direction vector
    u = x1 - x2
    u_norm = np.linalg.norm(u)

    # Construct the new weak learner (rank-1 trace-1 matrix)
    Z_new = np.outer(u, u) / (u_norm**2)
    #Z_new = Z_new.max() - Z_new  # Invert the matrix to decrease the distance
    print(Z_new)

    # Update the inverse covariance matrix
    S_inv_new = S_inv + alpha * Z_new

    return S_inv_new

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x1, x2, S_inv):
    diff = x1 - x2
    return np.sqrt(diff @ S_inv @ diff)


# Generate random data points
np.random.seed(42)
x1 = np.random.rand(3)
x2 = np.random.rand(3)

# Initial inverse covariance matrix (identity matrix)
S_inv_initial = np.eye(3)

# Calculate initial Mahalanobis distance
initial_distance = mahalanobis_distance(x1, x2, S_inv_initial)

# Update S^-1 with a new weak learner to decrease distance
alpha = 1.0  # Weight for the weak learner
S_inv_updated = update_metric_decrease_distance(S_inv_initial, x1, x2, alpha)
S_inv_updated_orth = update_metric_decrease_distance_orth(S_inv_initial, x1, x2, alpha)
#S_inv_updated = cov2corr(S_inv_updated)
#S_inv_updated_orth = cov2corr(S_inv_updated_orth)

# Calculate new Mahalanobis distance
assert is_pos_def(S_inv_updated)
assert is_pos_def(S_inv_updated_orth)
new_distance = mahalanobis_distance(x1, x2, S_inv_updated)
new_distance_orth = mahalanobis_distance(x1, x2, S_inv_updated_orth)
# Print the results
print("Initial Mahalanobis distance:", initial_distance)
print("New Mahalanobis distance after adding weak learner:", new_distance)
print("New Mahalanobis distance after adding weak learner orthogonal:", new_distance_orth)