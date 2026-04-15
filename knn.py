"""
knn.py — K-Nearest Neighbors from scratch (numpy only).

Supports:
  - Classification (majority vote)
  - Regression (mean of neighbors)
  - Distance metrics: euclidean, manhattan, minkowski
  - Weighted voting by inverse distance
  - sklearn-compatible API: fit / predict / score

The algorithm:
  Training  = just store X, y (lazy learner — no actual fitting)
  Inference = for each query point, find k closest training points,
              return majority class (classification) or mean value (regression)

Time complexity:
  Brute force: O(N * D) per query where N=training size, D=features
  This is why KNN doesn't scale — at N=1M, D=100, each prediction is 100M ops.
  Production fix: KD-trees (sklearn default) or ball trees for low-D data.
"""

import numpy as np
from collections import Counter
from typing_extensions import Literal


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    metric : str
        Distance metric. One of 'euclidean', 'manhattan', 'minkowski'.
    p : float
        Power for Minkowski distance. p=2 → euclidean, p=1 → manhattan.
    weights : str
        'uniform' = all neighbors vote equally.
        'distance' = closer neighbors get higher weight (1/dist).
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: float = 2.0,
        weights: Literal['uniform', 'distance'] = 'uniform',
    ):
        self.k       = k
        self.metric  = metric
        self.p       = p
        self.weights = weights
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        'Fit' = memorize training data.
        KNN is a lazy learner — no computation happens here.
        All the work is deferred to predict time.
        """
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray):
        distances = self._compute_distances(x)             # (N,)
        k_idx     = np.argsort(distances)[:self.k]         # indices of k nearest
        k_labels  = self._y_train[k_idx]
        k_dists   = distances[k_idx]

        if self.weights == 'uniform':
            # Simple majority vote
            return Counter(k_labels).most_common(1)[0][0]
        else:
            # Weighted vote: weight = 1 / distance
            # Handle exact matches (dist=0) → assign infinite weight
            weights = np.where(k_dists == 0, 1e10, 1.0 / k_dists)
            weight_per_class = {}
            for label, w in zip(k_labels, weights):
                weight_per_class[label] = weight_per_class.get(label, 0) + w
            return max(weight_per_class, key=weight_per_class.get)

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized distance computation between query point x and all training points.

        Broadcasting:
          self._X_train shape: (N, D)
          x shape:             (D,)
          diff shape:          (N, D)  ← x is broadcast across rows
        """
        diff = self._X_train - x          # (N, D)

        if self.metric == 'euclidean':
            return np.sqrt((diff ** 2).sum(axis=1))
        elif self.metric == 'manhattan':
            return np.abs(diff).sum(axis=1)
        elif self.metric == 'minkowski':
            return (np.abs(diff) ** self.p).sum(axis=1) ** (1.0 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score."""
        return float(np.mean(self.predict(X) == y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.
        Shape: (n_samples, n_classes)
        """
        X = np.array(X, dtype=float)
        probs = []
        for x in X:
            distances = self._compute_distances(x)
            k_idx    = np.argsort(distances)[:self.k]
            k_labels = self._y_train[k_idx]
            k_dists  = distances[k_idx]

            if self.weights == 'uniform':
                weights = np.ones(self.k)
            else:
                weights = np.where(k_dists == 0, 1e10, 1.0 / k_dists)

            prob_row = []
            for cls in self.classes_:
                mask = k_labels == cls
                prob_row.append(weights[mask].sum())
            prob_row = np.array(prob_row)
            probs.append(prob_row / prob_row.sum())
        return np.array(probs)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.

    Prediction = weighted or unweighted mean of k nearest neighbors' targets.
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: float = 2.0,
        weights: Literal['uniform', 'distance'] = 'uniform',
    ):
        self.k       = k
        self.metric  = metric
        self.p       = p
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> float:
        diff      = self._X_train - x
        distances = np.sqrt((diff ** 2).sum(axis=1))
        k_idx     = np.argsort(distances)[:self.k]
        k_targets = self._y_train[k_idx]
        k_dists   = distances[k_idx]

        if self.weights == 'uniform':
            return k_targets.mean()
        else:
            weights = np.where(k_dists == 0, 1e10, 1.0 / k_dists)
            return np.average(k_targets, weights=weights)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot
