"""
decision_tree.py — CART Decision Tree from scratch (numpy only).

Implements the Classification and Regression Trees (CART) algorithm
(Breiman et al., 1984) — the same algorithm used inside sklearn, XGBoost,
LightGBM, and CatBoost.

Understanding this deeply is essential because every boosting algorithm
is just an ensemble of these trees.

Key concepts implemented:
  - Greedy recursive binary splitting
  - Gini impurity (classification)
  - Entropy / information gain (classification)
  - MSE reduction (regression)
  - Max depth, min samples split, min samples leaf stopping criteria
  - Feature subsampling (used by Random Forest)
  - sklearn-compatible API
"""

import numpy as np
from typing_extensions import Optional, Literal


# ─────────────────────────────────────────────
# IMPURITY FUNCTIONS
# ─────────────────────────────────────────────

def gini_impurity(y: np.ndarray) -> float:
    """
    Gini impurity: 1 - sum(p_k^2)

    Measures how often a randomly chosen element would be incorrectly
    labeled if randomly labeled according to the class distribution.

    Range: [0, 1 - 1/K] where K = number of classes.
    Pure node (all same class): Gini = 0.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """
    Shannon entropy: -sum(p_k * log2(p_k))

    Measures information content / uncertainty in the labels.
    Pure node: entropy = 0. Maximally mixed: entropy = log2(K).
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Avoid log(0) — masked where p=0 (contributes 0 to sum anyway)
    return -np.sum(probs * np.log2(probs + 1e-12))


def mse_impurity(y: np.ndarray) -> float:
    """
    MSE impurity for regression: var(y)
    A pure node has variance 0.
    """
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


# ─────────────────────────────────────────────
# TREE NODE
# ─────────────────────────────────────────────

class Node:
    """
    A single node in the decision tree.

    Internal nodes have: feature_idx, threshold, left, right
    Leaf nodes have: value (prediction)
    """
    __slots__ = [
        'feature_idx', 'threshold', 'left', 'right',
        'value', 'impurity', 'n_samples', 'depth'
    ]

    def __init__(self):
        self.feature_idx : Optional[int]   = None
        self.threshold   : Optional[float] = None
        self.left        : Optional[Node]  = None
        self.right       : Optional[Node]  = None
        self.value       : Optional[float] = None   # set for leaf nodes
        self.impurity    : float           = 0.0
        self.n_samples   : int             = 0
        self.depth       : int             = 0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ─────────────────────────────────────────────
# DECISION TREE BASE
# ─────────────────────────────────────────────

class _DecisionTreeBase:
    """
    Base class shared by classifier and regressor.

    CART Algorithm (recursive binary splitting):
      1. For each feature, for each unique value as threshold:
           Split data into left (≤ threshold) and right (> threshold)
           Compute weighted impurity of the split
      2. Choose the feature + threshold that minimizes weighted impurity
      3. Recurse on left and right subsets
      4. Stop when max_depth reached, too few samples, or no improvement

    Time complexity: O(N * D * log N) per tree
      - N samples, D features
      - For each node: try D features × N thresholds
      - Tree depth is O(log N) on average
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,     # for Random Forest subsampling
        criterion: str = 'gini',
        random_state: Optional[int] = None,
    ):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.criterion         = criterion
        self.random_state      = random_state
        self.root_             = None
        self.n_features_in_    = None
        self.feature_importances_ = None
        self._rng = np.random.RandomState(random_state)

    def _impurity_fn(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]
        self._importance_sum = np.zeros(X.shape[1])

        self.root_ = self._build(X, y, depth=0)

        # Normalize feature importances
        total = self._importance_sum.sum()
        self.feature_importances_ = (
            self._importance_sum / total if total > 0
            else np.zeros(X.shape[1])
        )
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.impurity  = self._impurity_fn(y)
        node.depth     = depth

        # ── Stopping criteria ──────────────────────────────────────
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or node.impurity == 0.0):
            node.value = self._leaf_value(y)
            return node

        # ── Find best split ────────────────────────────────────────
        best = self._best_split(X, y)

        if best is None:
            node.value = self._leaf_value(y)
            return node

        feat, thresh, left_mask = best
        right_mask = ~left_mask

        # Enforce min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = self._leaf_value(y)
            return node

        # Track impurity decrease for feature importance
        # Importance = (N/N_total) * (impurity - N_L/N * imp_L - N_R/N * imp_R)
        N = len(y)
        imp_decrease = (
            node.impurity
            - (left_mask.sum() / N)  * self._impurity_fn(y[left_mask])
            - (right_mask.sum() / N) * self._impurity_fn(y[right_mask])
        )
        self._importance_sum[feat] += imp_decrease * N

        node.feature_idx = feat
        node.threshold   = thresh
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """
        Try all features and thresholds; return the split with lowest weighted impurity.

        For each feature:
          Sort by feature value → thresholds are midpoints between consecutive unique values
          Compute impurity reduction for each threshold

        This is O(N * D) which is the bottleneck for large datasets.
        XGBoost optimizes this with histogram binning (approximate split finding).
        """
        best_gain    = -np.inf
        best_feature = None
        best_thresh  = None
        best_mask    = None
        parent_imp   = self._impurity_fn(y)
        N            = len(y)

        # Feature subsampling (used by Random Forest / Gradient Boosting)
        n_features = X.shape[1]
        if self.max_features is not None:
            feat_indices = self._rng.choice(
                n_features,
                size=min(self.max_features, n_features),
                replace=False
            )
        else:
            feat_indices = np.arange(n_features)

        for feat in feat_indices:
            x_feat = X[:, feat]
            # Use unique sorted values as candidate thresholds
            thresholds = np.unique(x_feat)
            if len(thresholds) == 1:
                continue

            # Midpoints between consecutive unique values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for thresh in thresholds:
                left_mask  = x_feat <= thresh
                right_mask = ~left_mask

                n_left  = left_mask.sum()
                n_right = right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue

                # Weighted impurity after split
                imp_left  = self._impurity_fn(y[left_mask])
                imp_right = self._impurity_fn(y[right_mask])
                weighted  = (n_left * imp_left + n_right * imp_right) / N

                # Information gain = parent impurity - weighted child impurity
                gain = parent_imp - weighted

                if gain > best_gain:
                    best_gain    = gain
                    best_feature = feat
                    best_thresh  = thresh
                    best_mask    = left_mask

        if best_gain <= 0:
            return None
        return best_feature, best_thresh, best_mask

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    def _traverse(self, x: np.ndarray, node: Node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def get_depth(self) -> int:
        return self._depth(self.root_)

    def _depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))


# ─────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────

class DecisionTreeClassifier(_DecisionTreeBase):
    """
    CART Decision Tree for classification.

    Splitting criteria: 'gini' (default) or 'entropy'
    Leaf prediction: majority class
    """

    def __init__(self, criterion='gini', **kwargs):
        super().__init__(criterion=criterion, **kwargs)

    def _impurity_fn(self, y):
        return gini_impurity(y) if self.criterion == 'gini' else entropy(y)

    def _leaf_value(self, y):
        return Counter_mode(y)

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        return super().fit(X, y, **kwargs)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def predict_proba(self, X):
        # For a pure decision tree, leaf gives 0/1 probabilities
        # (useful for gradient boosting which works in probability space)
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            proba[preds == cls, i] = 1.0
        return proba


# ─────────────────────────────────────────────
# REGRESSOR
# ─────────────────────────────────────────────

class DecisionTreeRegressor(_DecisionTreeBase):
    """
    CART Decision Tree for regression.

    Splitting criterion: MSE reduction (variance reduction)
    Leaf prediction: mean of target values in leaf
    """

    def __init__(self, **kwargs):
        super().__init__(criterion='mse', **kwargs)

    def _impurity_fn(self, y):
        return mse_impurity(y)

    def _leaf_value(self, y):
        return float(y.mean())

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def Counter_mode(y):
    """Return the most common element in y."""
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]
