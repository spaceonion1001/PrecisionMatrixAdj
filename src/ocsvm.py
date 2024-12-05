import numpy as np
import scipy
from numba import njit
from numba_progress import ProgressBar
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
import cvxpy as cp
import cvxopt

class OCSVMMix:
    def __init__(self, v=0.1, kernel_approx=False, gamma=0.5, n_features=100):
        self.v = v
        self.w = None
        self.rho = None
        self.xi = None
        self.C = None
        self.labeled_indices = []
        self.kernel_approx = kernel_approx
        self.n_features_approx = n_features
        if kernel_approx:
            self.kernel_function = RBFKernelApproximation(gamma=gamma, n_features=n_features)

    def fit(self, X, y):
        self.data = X
        self.y = y
        n_samples, n_features = X.shape
        self.n = n_samples
        if self.kernel_approx:
            self.w = cp.Variable(self.n_features_approx)
        else:
            self.w = cp.Variable(n_features)
        self.rho = cp.Variable()
        self.xi = cp.Variable(n_samples, nonneg=True)
        self.C = cp.Parameter(nonneg=True)

        # Define the objective function
        objective = cp.Maximize(self.rho - (1 / (self.v * self.n)) * cp.sum(self.xi))

        # Define the constraints
        if not self.kernel_approx:
            constraints = [
                self.w @ X[i, :] >= self.rho - self.xi[i] for i in range(self.n)  # w * x_i >= rho - xi_i for each i
            ]
        else:
            self.X_transformed = self.kernel_function.fit_transform(X)
            constraints = [
                self.w @ self.X_transformed[i, :] >= self.rho - self.xi[i] for i in range(self.n)  # w * x_i >= rho - xi_i for each i
            ]
        constraints += [cp.norm(self.w) <= 1, cp.sum(self.w) == 1, self.w >= 0]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

    def decision_function(self, X):
        if self.kernel_approx:
            X = self.kernel_function.transform(X)
        return np.dot(X, self.w.value) - self.rho.value
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def add_labeled_example(self, indx):
        """
        Add a labeled example to the model.

        Parameters:
        x: ndarray of shape (n_features,), new labeled example
        label: int, 1 for inlier and -1 for outlier
        """
        # if label not in [1, -1]:
        #     raise ValueError("Label must be 1 (inlier) or -1 (outlier).")
        self.labeled_indices.append(indx)
        #self.labels.append(label)

        # Update the constraints
        self.C.value = 2.0
        n_unlabeled = self.n - len(self.labeled_indices)
        n_labeled = len(self.labeled_indices)
        self.xi_labeled = cp.Variable(n_labeled, nonneg=True)
        self.xi_unlabeled = cp.Variable(n_unlabeled, nonneg=True)
        X_labeled = self.data[self.labeled_indices, :]
        X_unlabeled = np.delete(self.data, self.labeled_indices, axis=0)
        if self.kernel_approx:
            X_labeled = self.kernel_function.transform(X_labeled)
            X_unlabeled = self.kernel_function.transform(X_unlabeled)
        y_labeled = self.y[self.labeled_indices]
        objective = cp.Maximize(
            self.rho - (1 / (self.v * n_unlabeled)) * cp.sum(self.xi_unlabeled) - self.C * cp.sum(self.xi_labeled)
        )
         # Constraints for unlabeled data
        constraints = [
            X_unlabeled @ self.w >= self.rho - self.xi_unlabeled,
            self.xi_unlabeled >= 0
        ]

        # Constraints for labeled data
        constraints += [
            *(y_labeled[j] * (X_labeled[j] @ self.w) >= 1 - self.xi_labeled[j] for j in range(n_labeled)),
            self.xi_labeled >= 0
        ]
        

        constraints += [cp.norm(self.w) <= 1, cp.sum(self.w) == 1, self.w >= 0]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()


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
    

class OCSVMCVXPrimalMinimization:
    def __init__(self, nu=0.1):
        """
        Initialize the One-Class SVM model.

        Parameters:
        nu: float, regularization parameter (0 < nu ≤ 1)
        """
        self.nu = nu
        self.w = None
        self.rho = None

    def fit(self, X):
        """
        Fit the One-Class SVM model to the data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Variables
        w = cp.Variable(n_features)
        rho = cp.Variable()
        xi = cp.Variable(n_samples, nonneg=True)

        # Objective
        objective = cp.Minimize((1 / (self.nu * n_samples)) * cp.sum(xi) - rho)

        # Constraints
        constraints = [X @ w >= rho - xi, xi >= 0]
        constraints += [cp.norm(w, 2) <= 1]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization did not converge")

        self.w = w.value
        self.rho = rho.value

    def decision_function(self, X):
        """
        Compute the decision scores for input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        scores: ndarray of shape (n_samples,)
        """
        return X @ self.w - self.rho

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        labels: ndarray of shape (n_samples,), 1 for inliers, -1 for outliers
        """
        scores = self.decision_function(X)
        return np.sign(scores)
    
class OCSVMCVXPrimalRad:
    # radius formulation
    def __init__(self, data, v=0.1, kernel_approx=True):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.radius = cp.Variable()
        self.xi = cp.Variable(self.n, nonneg=True)
        self.slack_weights = np.ones(self.n)

        self.hard_constraint_indices = []
        self.labels = []
        self.kernel_function = None
        if kernel_approx:
            self.kernel_function = RBFKernelApproximation(gamma=0.5, n_features=100)

        # define the objective function
        self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * cp.sum(self.xi))
        # define the constraints
        if kernel_approx:
            self.center = cp.Variable(self.kernel_function.n_features)
            self.constraints = [
                cp.norm(self.kernel_function.fit_transform(self.data)[i, :] - self.center, 2)**2 <= self.radius + self.xi[i] for i in range(self.n)
            ]
        else:
            self.center = cp.Variable(data.shape[1])
            self.constraints = [
                cp.norm(self.data[i, :] - self.center, 2)**2 <= self.radius + self.xi[i] for i in range(self.n)
            ]

    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve(solver=cp.SCS)

        return self.center.value, self.radius.value, self.xi.value

    def decision_function(self, x):
        if self.kernel_function:
            x = self.kernel_function.transform(x)
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
        self.slack_weights[index] = 5
        self.slack_weights /= 2
        self.slack_weights[self.hard_constraint_indices] *= 5

        weighted_slack = cp.sum(cp.multiply(self.slack_weights, self.xi))

        #soft_constraint = cp.norm(self.data[index, :] - self.center, 2)**2 <= self.radius + self.xi[index]
        
        # define the objective function
        self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * weighted_slack)
        #self.objective = cp.Minimize(self.radius + (1 / (self.v * self.n)) * cp.sum(self.xi))
        # define the constraints
        if self.kernel_function:
            self.constraints = [
                cp.norm(self.kernel_function.transform(self.data)[i, :] - self.center, 2)**2 <= self.radius + self.xi[i] for i in range(self.n)
            ]
        else:
            self.constraints = [
                cp.norm(self.data[i, :] - self.center, 2)**2 <= (self.radius + self.xi[i]) for i in range(self.n)
            ]

        # self.constraints = []
        # for i in range(self.n):
        #     aux_dist = cp.norm(self.data[i, :] - self.center, 2)**2
        #     aux_dist_root = cp.norm(self.data[i, :] - self.center, 2)
        #     if i in self.hard_constraint_indices:
        #         clabel = self.labels[self.hard_constraint_indices.index(i)]
        #         if clabel == 1: # inlier
        #             #print("I'M AN INLIER")
        #             constraint = aux_dist <= self.radius + self.xi[i]
        #         elif clabel == -1:          # outlier
        #             #print("I'M AN OUTLIER")
        #             constraint = -aux_dist_root <= -cp.sqrt(self.radius + 1e-6 + self.xi[i])
        #     else:
        #         constraint = aux_dist <= self.radius + self.xi[i]
        #     self.constraints.append(constraint)
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
    def __init__(self, data, v, kernel='rbf', gamma=1.0):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.m = data.shape[1]

        self.kernel = kernel
        self.gamma = gamma
        #self.K = data @ data.T
        self.K = self._kernel_function(data, data)

        self.hard_constraint_indices = set()
        # Define dual variables
        self.alpha = cp.Variable(self.n)

        # Objective function for the dual
        self.objective = cp.Minimize(cp.quad_form(self.alpha, cp.Parameter(shape=self.K.shape, value=self.K, PSD=True)) - cp.sum(cp.multiply(self.alpha, np.diag(self.K))))
        self.constraints = [self.alpha >= 0, self.alpha <= 1 / (self.v * self.n), cp.sum(self.alpha) == 1]

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type")

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
        #K_test_train = x @ self.data.T
        K_test_train = self._kernel_function(x, self.data)
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
    def __init__(self, data, v, kernel='rbf', gamma=1.0):
        self.data = data
        self.n = data.shape[0]
        self.v = v
        self.alpha = cp.Variable(self.n)
        self.kernel = kernel
        self.gamma = gamma
        #self.K = data @ data.T
        self.K = self._kernel_function(data, data)
        self.objective = cp.Minimize(0.5 * cp.quad_form(self.alpha, cp.Parameter(shape=self.K.shape, value=self.K, PSD=True)))

        # Define the constraints
        self.constraints = [
            self.alpha >= 0,                  # Each alpha_i >= 0
            self.alpha <= 1 / (self.v * self.n),        # Each alpha_i <= 1/(v*n)
            cp.sum(self.alpha) == 1,           # Sum of all alpha_i equals 1
        ]

        self. labels = np.array([None] * self.n) 

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type")
        
    def solve(self):
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve()

        return self.alpha.value
    
    def decision_function(self, x):
        K_test = self._kernel_function(x, self.data)

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

class SemiSupervisedOneClassSVM:
    def __init__(self, nu=0.1, kernel='rbf', gamma=1.0):
        """
        Initialize the Semi-Supervised One-Class SVM.

        Parameters:
        nu: float, regularization parameter (0 < nu ≤ 1)
        kernel: str, kernel type ('linear', 'rbf')
        gamma: float, parameter for RBF kernel
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.X_unlabeled = None
        self.labeled_data = []  # To store labeled data (x, label)
        
    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X_unlabeled):
        """
        Fit the One-Class SVM to the unlabeled data.

        Parameters:
        X_unlabeled: ndarray of shape (n_samples, n_features)
        """
        self.X_unlabeled = X_unlabeled
        n_samples = X_unlabeled.shape[0]

        # Kernel matrix
        K = self._kernel_function(X_unlabeled, X_unlabeled)

        # Define optimization variables
        alpha = cp.Variable(n_samples, nonneg=True)

        # Define the problem
        objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.Parameter(shape=K.shape, value=K, PSD=True)))
        constraints = [cp.sum(alpha) == self.nu * n_samples, alpha >= 0]
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization failed")

        self.alpha = alpha.value
        self.support_vectors = X_unlabeled[self.alpha > 1e-6]
        self.support_vector_indices = np.where(self.alpha > 1e-6)[0]
        self.bias = np.mean(K[self.support_vector_indices, :] @ self.alpha - 1)

    def add_labeled_data(self, x, label):
        """
        Add labeled data and refine the model dynamically.

        Parameters:
        x: ndarray of shape (n_features,)
        label: int, 1 for normal and -1 for anomaly
        """
        self.labeled_data.append((x, label))

        # Update the decision boundary using labeled data
        if len(self.labeled_data) > 0:
            X_labeled = np.array([d[0] for d in self.labeled_data])
            y_labeled = np.array([d[1] for d in self.labeled_data])

            # Kernel matrices
            K_ul = self._kernel_function(self.X_unlabeled, X_labeled)
            K_ll = self._kernel_function(X_labeled, X_labeled)
            
            # Incorporate labeled data constraints
            n_samples = self.X_unlabeled.shape[0]
            m_labeled = len(y_labeled)
            alpha = cp.Variable(n_samples, nonneg=True)
            beta = cp.Variable(m_labeled)

            # Redefine optimization problem
            objective = cp.Maximize(
                cp.sum(alpha) - 0.5 * cp.quad_form(alpha, self._kernel_function(self.X_unlabeled, self.X_unlabeled))
                - cp.quad_form(beta, K_ll)
            )

            constraints = [
                cp.sum(alpha) == self.nu * n_samples,
                alpha >= 0,
                y_labeled * (K_ul.T @ alpha + beta) >= 1  # Labeled constraints
            ]

            problem = cp.Problem(objective, constraints)

            problem.solve()

            if problem.status != cp.OPTIMAL:
                raise ValueError("Optimization failed with labeled data")

            self.alpha = alpha.value
            self.beta = beta.value

    def decision_function(self, X):
        """
        Compute the decision function for the input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        scores: ndarray of shape (n_samples,)
        """
        K = self._kernel_function(X, self.X_unlabeled)
        decision_scores = K @ self.alpha - self.bias

        if len(self.labeled_data) > 0:
            K_labeled = self._kernel_function(X, np.array([d[0] for d in self.labeled_data]))
            decision_scores += K_labeled @ self.beta

        return decision_scores

    def predict(self, X):
        """
        Predict labels for the input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        labels: ndarray of shape (n_samples,), 1 for normal, -1 for anomaly
        """
        return np.sign(self.decision_function(X))
    

class RBFKernelApproximation:
    def __init__(self, gamma=1.0, n_features=100):
        """
        Initialize the RBF kernel approximation using Random Fourier Features.

        Parameters:
        gamma: float, RBF kernel parameter
        n_features: int, number of random Fourier features
        """
        self.gamma = gamma
        self.n_features = n_features
        self.omega = None
        self.bias = None

    def fit(self, X):
        np.random.seed(42)
        """
        Fit the random Fourier features to the input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)
        """
        n_features = X.shape[1]
        self.omega = np.random.normal(0, np.sqrt(2 * self.gamma), size=(n_features, self.n_features))
        self.bias = np.random.uniform(0, 2 * np.pi, size=self.n_features)

    def transform(self, X):
        """
        Transform the input data using the random Fourier feature map.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        Z: ndarray of shape (n_samples, n_features)
        """
        projection = X @ self.omega + self.bias
        Z = np.sqrt(2 / self.n_features) * np.cos(projection)
        return Z

    def fit_transform(self, X):
        """
        Fit the random Fourier features and transform the input data.

        Parameters:
        X: ndarray of shape (n_samples, n_features)

        Returns:
        Z: ndarray of shape (n_samples, n_features)
        """
        self.fit(X)
        return self.transform(X)

