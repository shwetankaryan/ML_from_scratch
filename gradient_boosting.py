"""
gradient_boosting.py — Gradient Boosting from scratch (numpy only).

Implements the algorithm from Friedman (2001) "Greedy Function Approximation:
A Gradient Boosting Machine" — the paper that unified all boosting methods
under a single framework.

Core idea:
  Instead of fitting residuals directly (like AdaBoost), Gradient Boosting
  fits trees to the *negative gradient* of the loss function.

  For MSE loss: negative gradient = residuals (y - y_hat)
  For log loss: negative gradient = y - sigmoid(y_hat)   (probability residuals)

  This means GB is doing gradient descent in function space:
    F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
  where h_m is a tree fit to the negative gradient.

Key hyperparameters:
  n_estimators:  number of trees (more = lower bias, higher variance + time)
  learning_rate: shrinkage factor (lower = need more trees, but often better)
  max_depth:     tree depth (shallow trees = high bias, prevents overfitting)
  subsample:     fraction of data per tree (stochastic GB, reduces variance)
  max_features:  features per split (additional regularization)
"""

import numpy as np
from typing_extensions import Literal, Optional
from decision_tree import DecisionTreeRegressor


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

class MSELoss:
    """Mean Squared Error — used for regression."""

    @staticmethod
    def negative_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Negative gradient of MSE w.r.t. predictions.
        MSE = mean((y - F)^2)
        dL/dF = -(y - F) → negative gradient = (y - F) = residuals
        """
        return y - y_pred

    @staticmethod
    def initial_prediction(y: np.ndarray) -> float:
        """
        Best constant prediction = mean(y).
        This minimizes MSE before any trees are added.
        """
        return y.mean()

    @staticmethod
    def loss(y, y_pred):
        return np.mean((y - y_pred) ** 2)


class LogLoss:
    """
    Binary cross-entropy — used for binary classification.
    Model outputs raw log-odds (logits), not probabilities.
    """

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def negative_gradient(cls, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        y_pred here is log-odds (before sigmoid).
        p = sigmoid(y_pred)
        negative gradient = y - p  (probability residuals)
        """
        return y - cls.sigmoid(y_pred)

    @staticmethod
    def initial_prediction(y: np.ndarray) -> float:
        """
        Best constant = log(mean(y) / (1 - mean(y)))  → log-odds of base rate
        """
        p = y.mean()
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @classmethod
    def loss(cls, y, y_pred):
        p = cls.sigmoid(y_pred)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ─────────────────────────────────────────────
# GRADIENT BOOSTING REGRESSOR
# ─────────────────────────────────────────────

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (MSE loss).

    Algorithm:
      1. Initialize F_0(x) = mean(y)
      2. For m = 1 to M:
           a. Compute pseudo-residuals: r_i = -[dL/dF(x_i)]  = y_i - F_{m-1}(x_i)
           b. Fit a regression tree h_m to r
           c. Update: F_m(x) = F_{m-1}(x) + lr * h_m(x)
      3. Return F_M(x)
    """

    def __init__(
        self,
        n_estimators:  int   = 100,
        learning_rate: float = 0.1,
        max_depth:     int   = 3,
        min_samples_split: int = 2,
        subsample:     float = 1.0,    # stochastic GB if < 1
        max_features:  Optional[int] = None,
        random_state:  Optional[int] = None,
    ):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.subsample         = subsample
        self.max_features      = max_features
        self.random_state      = random_state
        self._trees            = []
        self._F0               = None
        self._loss             = MSELoss()
        self.train_loss_       = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X   = np.array(X, dtype=float)
        y   = np.array(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        # Step 1: Initialize with best constant prediction
        self._F0 = self._loss.initial_prediction(y)
        F = np.full(len(y), self._F0)           # current predictions

        self._trees = []

        for m in range(self.n_estimators):

            # Step 2a: Compute pseudo-residuals (negative gradient)
            residuals = self._loss.negative_gradient(y, F)  # (N,)

            # Stochastic gradient boosting: subsample rows
            if self.subsample < 1.0:
                n_sub = max(1, int(len(X) * self.subsample))
                idx   = rng.choice(len(X), size=n_sub, replace=False)
            else:
                idx = np.arange(len(X))

            # Step 2b: Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features      = self.max_features,
                random_state      = rng.randint(0, 10000),
            )
            tree.fit(X[idx], residuals[idx])

            # Step 2c: Update predictions
            # Note: tree is fit on subsample but we update all predictions
            update = tree.predict(X)
            F += self.learning_rate * update

            self._trees.append(tree)
            self.train_loss_.append(self._loss.loss(y, F))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.full(len(X), self._F0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X: np.ndarray):
        """
        Generator yielding predictions after each tree is added.
        Useful for finding the optimal n_estimators (early stopping).
        """
        X = np.array(X, dtype=float)
        F = np.full(len(X), self._F0)
        yield F.copy()
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────
# GRADIENT BOOSTING CLASSIFIER
# ─────────────────────────────────────────────

class GradientBoostingClassifier:
    """
    Gradient Boosting for binary classification (log loss).

    Works identically to the regressor, but:
      - Initial prediction = log-odds of base rate
      - Pseudo-residuals = y - sigmoid(F)  (probability residuals)
      - Output of predict() is converted via sigmoid to probability,
        then thresholded at 0.5 for class prediction
    """

    def __init__(
        self,
        n_estimators:  int   = 100,
        learning_rate: float = 0.1,
        max_depth:     int   = 3,
        min_samples_split: int = 2,
        subsample:     float = 1.0,
        max_features:  Optional[int] = None,
        random_state:  Optional[int] = None,
    ):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.subsample         = subsample
        self.max_features      = max_features
        self.random_state      = random_state
        self._trees            = []
        self._F0               = None
        self._loss             = LogLoss()
        self.train_loss_       = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        X   = np.array(X, dtype=float)
        y   = np.array(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        self._F0 = self._loss.initial_prediction(y)
        F = np.full(len(y), self._F0)

        for m in range(self.n_estimators):
            residuals = self._loss.negative_gradient(y, F)

            if self.subsample < 1.0:
                n_sub = max(1, int(len(X) * self.subsample))
                idx   = rng.choice(len(X), size=n_sub, replace=False)
            else:
                idx = np.arange(len(X))

            tree = DecisionTreeRegressor(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features      = self.max_features,
                random_state      = rng.randint(0, 10000),
            )
            tree.fit(X[idx], residuals[idx])

            F += self.learning_rate * tree.predict(X)
            self._trees.append(tree)
            self.train_loss_.append(self._loss.loss(y, F))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.full(len(X), self._F0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        p = LogLoss.sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
