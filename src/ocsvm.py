import numpy as np
import scipy
from numba import njit
from numba_progress import ProgressBar
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
import cvxpy as cp
import cvxopt

class OCSVMCVXPrimal:
    def __init__(self, data, v=0.1):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.w = cp.Variable(data.shape[1])
        self.rho = cp.Variable()
        self.xi = cp.Variable(self.n, nonneg=True)

        # define the objective function
        self.objective = cp.Maximize(self.rho - (1 / (self.v * self.n)) * cp.sum(self.xi))
        # Define the constraints
        self.constraints = [
            self.w @ self.data[i, :] >= self.rho - self.xi[i] for i in range(self.n)  # w * x_i >= rho - xi_i for each i
        ]
        self.constraints += [
            cp.sum(self.w) == 1,  # Sum of w_i equals 1
            self.w >= 0           # w_i >= 0
        ]
    
    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve()

        return self.w.value, self.rho.value, self.xi.value
    
    def decision_function(self, x):
        decision_func = np.dot(np.array(self.w.value), x.T) - self.rho.value
        return decision_func

    def predict(self, x):
        return np.sign(self.decision_function(x))
    
class OCSVMCVXPrimalRad:
    # radius formulation
    def __init__(self, data, v=0.1):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.radius = cp.Variable()
        self.xi = cp.Variable(self.n, nonneg=True)
        self.center = cp.Variable(data.shape[1])
        self.slack_weights = np.ones(self.n)

        self.hard_constraint_indices = []
        self.labels = []

        # define the objective function
        self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * cp.sum(self.xi))
        # define the constraints
        self.constraints = [
            cp.norm(self.data[i, :] - self.center, 2)**2 <= self.radius + self.xi[i] for i in range(self.n)
        ]

    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve(solver=cp.SCS)

        return self.center.value, self.radius.value, self.xi.value

    def decision_function(self, x):
        distances_squared = np.sum((x - self.center.value) ** 2, axis=1)
        decision_values = self.radius.value - distances_squared

        return decision_values
    
    def predict(self, x):
        return np.sign(self.decision_function(x))
    
    def add_hard_constraint(self, index, label):
        """
        Add a hard margin constraint for a specific point, replacing its soft constraint.
        
        Parameters:
        - index: Index of the point to enforce a hard margin constraint.
        """
        print(f"Adding hard constraint for point {index} with label {label}.")
        if index in self.hard_constraint_indices:
            print(f"Point {index} already has a hard constraint.")
            return
        self.labels.append(label)
        self.hard_constraint_indices.append(index)

        self.slack_weights[index] = 10
        weighted_slack = cp.sum(cp.multiply(self.slack_weights, self.xi))

        #soft_constraint = cp.norm(self.data[index, :] - self.center, 2)**2 <= self.radius + self.xi[index]
        
        # define the objective function
        #self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * weighted_slack)
        self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * cp.sum(self.xi))
        # define the constraints
        self.constraints = [
            cp.norm(self.data[i, :] - self.center, 2)**2 <= (self.radius + self.xi[i]) for i in range(self.n)
        ]

        self.constraints = []
        for i in range(self.n):
            aux_dist = cp.norm(self.data[i, :] - self.center, 2)**2
            aux_dist_root = cp.norm(self.data[i, :] - self.center, 2)
            if i in self.hard_constraint_indices:
                clabel = self.labels[self.hard_constraint_indices.index(i)]
                if clabel == 1: # inlier
                    #print("I'M AN INLIER")
                    constraint = aux_dist <= self.radius + self.xi[i]
                elif clabel == -1:          # outlier
                    #print("I'M AN OUTLIER")
                    constraint = -aux_dist_root <= -cp.sqrt(self.radius + 1e-6 + self.xi[i])
            else:
                constraint = aux_dist <= self.radius + self.xi[i]
            self.constraints.append(constraint)
        self.constraints += [self.xi >= 0]


        # redfine objective function and slack variables
        # self.xi = cp.Variable(self.n-len(self.hard_constraint_indices), nonneg=True)
        # self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * cp.sum(self.xi))
        # xi_indices = [i for i in range(self.n) if i not in self.hard_constraint_indices]
        # xi_counter = 0

        # # soft constraints
        # for idx in xi_indices:
        #     print("XI INDEX")
        #     print(idx)
        #     aux_dist = cp.norm(self.data[idx, :] - self.center, 2)**2
        #     self.constraints.append(aux_dist <= self.radius + self.xi[xi_counter])
        #     xi_counter += 1

        # # hard constraints
        # for i in range(len(self.hard_constraint_indices)):
        #     print("HARD CONSTRAINT INDEX")
        #     print(self.hard_constraint_indices[i])
        #     clabel = self.labels[i]
        #     aux_dist = cp.norm(self.data[self.hard_constraint_indices[i], :] - self.center, 2)**2
        #     if clabel == 1: # inlier
        #         print("I'M AN INLIER")
        #         hard_constraint = aux_dist <= self.radius
        #     elif clabel == -1:          # outlier
        #         print("I'M AN OUTLIER")
        #         hard_constraint = aux_dist >= (self.radius + 1e-6)
        #     else:
        #         raise ValueError("Label must be 1 (inlier) or -1 (outlier).")
        #     self.constraints.append(hard_constraint)

        # # Update the optimization problem

        return self.solve()
    

class OCSVMCVXDualRad:
    # radius formulation
    def __init__(self, data, v=0.1):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.m = data.shape[1]

        self.K = data @ data.T

        self.hard_constraint_indices = set()
        # Define dual variables
        self.alpha = cp.Variable(self.n)

        # Objective function for the dual
        self.objective = cp.Minimize(cp.quad_form(self.alpha, cp.Parameter(shape=self.K.shape, value=self.K, PSD=True)) - cp.sum(cp.multiply(self.alpha, np.diag(self.K))))
        self.constraints = [self.alpha >= 0, self.alpha <= 1 / (self.v * self.n), cp.sum(self.alpha) == 1]

    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve()

        #self.radius_squared = np.sum(self.alpha.value * self.alpha.value * self.K)
        return self.alpha.value

    def decision_function(self, x):
        support_indices = np.where(self.alpha.value <= 1 / (self.v * self.n))[0]
        x_k = self.data[support_indices[0]]
        R_squared = (
            np.dot(x_k, x_k)  # ||x_k||^2
            - 2 * np.dot(self.alpha.value, self.data @ x_k)  # 2 * sum_i alpha_i <x_i, x_k>
            + np.sum(self.alpha.value[:, None] * self.alpha.value[None, :] * self.K)  # sum_{i,j} alpha_i * alpha_j * <x_i, x_j>
        )
        test_norms_squared = np.sum(x ** 2, axis=1)
        K_test_train = x @ self.data.T
        dual_term = 2 * np.dot(K_test_train, self.alpha.value)
        radius_term = np.sum(self.alpha.value[:, None] * self.alpha.value[None, :] * self.K)
        decision_values = test_norms_squared - dual_term + radius_term
        return decision_values - R_squared
    
    def predict(self, x):
        return np.sign(self.decision_function(x))
    
class OCSVMRadAlt:
    def __init__(self, v, n_features):
        """
        Initialize the One-Class SVM problem.

        Parameters:
        - v (float): Hyperparameter for controlling the trade-off between R^2 and the slack variables.
        - n_features (int): Number of features in the input data.
        """
        self.v = v
        self.n_features = n_features
        self.R = cp.Variable(nonneg=True)  # Radius (scalar)
        self.c = cp.Variable(n_features)  # Center (vector)
        self.xi = None  # Slack variables (to be initialized per data)
        self.problem = None  # CVXPY Problem instance
        self.constraints = []
        self.X = None

        self.hard_constraint_indices = []
        self.labels = []

    def fit(self, X):
        """
        Solve the optimization problem for the given data X.

        Parameters:
        - X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        - dict: Optimization results containing 'R', 'c', and 'xi'.
        """
        self.X = X
        n_samples = X.shape[0]
        self.xi = cp.Variable(n_samples, nonneg=True)  # Slack variables for each data point
        
        # Define the objective function
        objective = cp.Minimize(self.R + (1 / (self.v * n_samples)) * cp.sum(self.xi))
        
        # Define the constraints
        self.constraints = [
            cp.norm(X[i] - self.c)**2 <= self.R + self.xi[i] for i in range(n_samples)
        ] + [self.xi >= 0]
        
        # Define the problem
        self.problem = cp.Problem(objective, self.constraints)
        
        # Solve the problem
        self.problem.solve()

        return {
            "R": self.R.value,
            "c": self.c.value,
            "xi": self.xi.value,
            "status": self.problem.status
        }
    
    def update_constraints(self, index, label):
        """
        Update constraints to enforce a specific point in X being inside or outside the hypersphere.

        Parameters:
        - index (int): Index of the point in the original dataset X.
        - label (int): Desired status of the point, either 1 (inlier) or -1 (outlier).

        Returns:
        - None
        """
        if label not in [1, -1]:
            raise ValueError("Label must be 1 (inlier) or -1 (outlier).")
        print(f"Adding hard constraint for point {index} with label {label}.")
        if index in self.hard_constraint_indices:
            print(f"Point {index} already has a hard constraint.")
            return
        self.labels.append(label)
        self.hard_constraint_indices.append(index)
        
        # Remove the original constraint for this point
        self.constraints.pop(index)

        # Get the point from the dataset
        point = self.X[index]

        # Add the new hard constraint
        if label == 1:  # Inlier
            self.constraints.insert(index, cp.norm(point - self.c)**2 <= self.R)
        elif label == -1:  # Outlier
            self.constraints.insert(index, cp.norm(point - self.c)**2 >= self.R + 1e-6)
        
        # Update and solve the problem with new constraints
        objective = cp.Minimize(self.R + (1 / (self.v * self.X.shape[0])) * cp.sum(self.xi))
        self.problem = cp.Problem(objective, self.constraints)
        self.problem.solve()
    
    def predict(self, X):
        """
        Predict the labels for the given data X.

        Parameters:
        - X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted labels for each data point.
        """
        return np.sign(self.decision_function(X))
    
    def decision_function(self, X):
        """
        Calculate the decision function for the given data X.

        Parameters:
        - X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Decision function values for each data point.
        """
        return self.R.value - np.sum((X - self.c.value)**2, axis=1)

class OCSVMCVXDual:
    def __init__(self, data, v):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.alpha = cp.Variable(self.n)
        self.K = data @ data.T
        self.objective = cp.Minimize(0.5 * cp.quad_form(self.alpha, cp.Parameter(shape=self.K.shape, value=self.K, PSD=True)))

        # Define the constraints
        self.constraints = [
            self.alpha >= 0,                  # Each alpha_i >= 0
            self.alpha <= 1 / (self.v * self.n),        # Each alpha_i <= 1/(v*n)
            cp.sum(self.alpha) == 1,           # Sum of all alpha_i equals 1
        ]

        self. labels = np.array([None] * self.n) 

    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve()

        return self.alpha.value
    
    def decision_function(self, x):
        K_test = x @ self.data.T

        support_vector_indices = np.where((self.alpha.value > 1e-5) & (self.alpha.value < (1 / (self.v * self.n))))[0]

        # Select a support vector for calculating rho
        #s = support_vector_indices[0]
        #rho = np.sum(self.alpha.value * self.K[s, :])
        

        #rho = np.sum(self.alpha.value * K_test[s, :])
        #rho = np.sum(self.alpha.value * self.K_test)
        alpha_support = self.alpha.value[support_vector_indices]
        self.w = alpha_support.dot(self.data[support_vector_indices, :])
        rho = self.w.dot(self.data[support_vector_indices, :].T).sum()/len(support_vector_indices)
        #idx = int(np.argmin(self.alpha.value[support_vector_indices]))
        #K_support = self.K[support_vector_indices][:, support_vector_indices]
        #rho = self.alpha.value[support_vector_indices].dot(K_support[idx])
        #print("Calculated rho:", rho)
        decision_values = K_test @ self.alpha.value - rho

        return decision_values
    
    def predict(self, x):
        return np.sign(self.decision_function(x))
    
    def update_constraints(self, index, label):
         # Remove the old constraint on alpha[index]
        self.constraints = [c for c in self.constraints if self.alpha[index] not in c.variables()]
        
        # Add new constraints based on the label
        if label == 1:  # Inlier constraint
            constraints += [1e-5 <= self.alpha[index], self.alpha[index] <= 1 / (self.v * self.n)]
        elif label == -1:  # Outlier constraint
            constraints += [self.alpha[index] == 0]
        
        # Update the problem with new constraints
        return self.solve()


class OCSVMBoost:
    def __init__(self, data, m=15): # m is the number of weak learners
        self.data = data
        self.n = data.shape[0]
        self.m = m
        self.hypotheses = []
        self.beta = 0
        self.u = np.ones(self.n) / self.n
        self.H = np.zeros((self.n, self.m))

    def fit(self):
        total_m = 0
        for i in range(self.m):
            h = self.train_weak_learner() # find weak hypothesis using eq 10
            y_preds = h.predict(self.data)
            sum_y_preds = np.sum(self.u*y_preds)
            if sum_y_preds <= self.beta and i > 0:
                print()
                print("Sum y preds {} Beta {}".format(sum_y_preds, self.beta))
                break
            self.hypotheses.append(h)
            self.H[:, i] = y_preds
            total_m += 1
            # solve restricted master for new costs
            self.u, self.beta, self.lg_mults = self.solve_master()
        print("Returning Hypotheses and Lagrange Multipliers")
        return self.hypotheses, np.array(self.lg_mults)
    
    def train_weak_learner(self):
        clf = OneClassSVM(kernel='linear', nu=0.5, gamma='scale', shrinking=False)
        clf.fit(self.data, sample_weight=self.u)
        return clf
    
    def solve_master(self, v=0.5):
        curr_m = len(self.hypotheses)
        D = 1/(v*self.n)

        u = cp.Variable(self.n)
        beta = cp.Variable()

        # define constraints
        constraints = [
            cp.sum(u) == 1, # sum of u_i = 1
            u >= 0, # u_i >= 0
            u <= D # u_i <= D
        ]

        for j in range(curr_m):
            constraints.append(cp.sum(cp.multiply(u, self.H[:, j])) <= beta)

        objective = cp.Minimize(beta)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # print("\nLagrange multipliers (dual values) for the constraints:")
        # for i, constraint in enumerate(constraints):
        #     print(f"Constraint {i+1} dual value: {constraint.dual_value}")
        # print u and beta current values
        #print("\nCurrent u values: {}".format(u.value))
        print("Current beta value: {}".format(beta.value))
        lg_mults = [constraint.dual_value for constraint in constraints[3:]]
        return u.value, beta.value, lg_mults
    

class NaiveBoostedOneClassSVM:
    def __init__(self, n_estimators=10, nu=0.5, gamma='scale', max_samples=0.8):
        self.n_estimators = n_estimators
        self.nu = nu
        self.gamma = gamma
        self.max_samples = max_samples
        self.models = []
        self.model_weights = []

    def fit(self, X):
        # Train multiple OneClassSVM models on subsets of the data
        for j in range(self.n_estimators):
            print("Fitting model", j)
            # Bootstrap sampling to introduce diversity
            X_sample = resample(X, n_samples=int(self.max_samples * len(X)), random_state=42)
            model = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel='linear', shrinking=False)
            model.fit(X_sample)
            self.models.append(model)

        # Self-consistency weighting
        predictions = np.array([model.predict(X) for model in self.models])
        majority_vote = np.sign(np.sum(predictions, axis=0))
        
        # Calculate weights based on model self-consistency with the majority vote
        for prediction in predictions:
            consistency = np.mean(prediction == majority_vote)
            self.model_weights.append(consistency)

        # Normalize weights to sum to 1
        self.model_weights = np.array(self.model_weights) / np.sum(self.model_weights)

    def predict(self, X):
        print("Predicting")
        # Aggregate predictions using weighted voting
        weighted_predictions = np.zeros(len(X))
        
        for model, weight in zip(self.models, self.model_weights):
            weighted_predictions += weight * model.predict(X)
        
        # Final decision based on weighted majority voting
        return np.sign(weighted_predictions)

    def decision_function(self, X):
        # Aggregated decision function for anomaly scores
        weighted_scores = np.zeros(len(X))
        
        for model, weight in zip(self.models, self.model_weights):
            weighted_scores += weight * model.decision_function(X)
        
        return weighted_scores
    

def qp(P, q, A, b, C, verbose=True):
    # Gram matrix
    n = P.shape[0]
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    G = cvxopt.matrix(np.concatenate(
        [np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0))
    h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
    return np.ravel(solution['x'])

def ocsvm_solver(K, nu=0.1):
    n = len(K)
    P = K
    q = np.zeros(n)
    A = np.matrix(np.ones(n))
    b = 1.
    C = 1. / (nu * n)
    mu = qp(P, q, A, b, C, verbose=False)
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support

def compute_rho(K, mu_support, idx_support):
    # TODO
    index = int(np.argmin(mu_support))
    K_support = K[idx_support][:, idx_support]
    rho = mu_support.dot(K_support[index])
    return rho
# # Fit the boosted OneClassSVM
# boosted_ocsvm = BoostedOneClassSVM(n_estimators=10, nu=0.1, gamma=0.1, max_samples=0.8)
# boosted_ocsvm.fit(X)

# # Predict anomalies (-1 for outliers, +1 for inliers)
# predictions = boosted_ocsvm.predict(X)
# print("Predictions:", predictions)

# # Anomaly scores (higher scores indicate more likely inliers)
# scores = boosted_ocsvm.decision_function(X)
# print("Anomaly scores:", scores)
