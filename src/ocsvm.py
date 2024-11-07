import numpy as np
import scipy
from numba import njit
from numba_progress import ProgressBar
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
import cvxpy as cp

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

# # Fit the boosted OneClassSVM
# boosted_ocsvm = BoostedOneClassSVM(n_estimators=10, nu=0.1, gamma=0.1, max_samples=0.8)
# boosted_ocsvm.fit(X)

# # Predict anomalies (-1 for outliers, +1 for inliers)
# predictions = boosted_ocsvm.predict(X)
# print("Predictions:", predictions)

# # Anomaly scores (higher scores indicate more likely inliers)
# scores = boosted_ocsvm.decision_function(X)
# print("Anomaly scores:", scores)
