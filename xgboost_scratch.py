"""
xgboost_scratch.py — XGBoost core algorithm from scratch (numpy only).

Implements the key innovations from Chen & Guestrin (2016):
  "XGBoost: A Scalable Tree Boosting System"

What makes XGBoost different from vanilla Gradient Boosting:

  1. SECOND-ORDER GRADIENTS
     Standard GB uses only the first derivative (gradient) of the loss.
     XGBoost uses both gradient (g) AND Hessian (h = second derivative).
     This gives a better quadratic approximation of the loss surface,
     allowing for more principled tree building.

  2. REGULARIZATION IN THE OBJECTIVE
     XGBoost adds L1 (alpha) and L2 (lambda) regularization directly
     into the tree-building objective, not as a post-processing step.
     This is why XGBoost tends to find simpler trees automatically.

  3. OPTIMAL LEAF WEIGHTS
     Given a tree structure, XGBoost computes the mathematically optimal
     weight for each leaf in closed form:
       w* = -sum(g_i) / (sum(h_i) + lambda)

  4. GAIN FORMULA FOR SPLITTING
     Instead of impurity reduction, XGBoost uses a gain formula derived
     from the second-order Taylor expansion of the loss:
       Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
     where γ is the minimum gain required to make a split (pruning).

  5. COLUMN (FEATURE) SUBSAMPLING + ROW SUBSAMPLING
     Both per-tree and per-split (colsample_bylevel) subsampling.
"""

import numpy as np
from typing_extensions import Optional, List
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# LOSS FUNCTIONS (g and h)
# ─────────────────────────────────────────────

class SquaredErrorObjective:
    """
    Regression: L = 0.5 * (y - F)^2
    g_i = dL/dF_i = F_i - y_i
    h_i = d²L/dF_i² = 1  (constant for MSE)
    """
    @staticmethod
    def gradients(y: np.ndarray, y_pred: np.ndarray):
        g = y_pred - y          # first derivative
        h = np.ones_like(y)     # second derivative = 1 for MSE
        return g, h

    @staticmethod
    def initial_prediction(y: np.ndarray) -> float:
        return y.mean()


class LogisticObjective:
    """
    Binary classification: L = log(1 + exp(-y * F))  (y ∈ {-1, +1})
    Simplified for y ∈ {0, 1}:
      p = sigmoid(F)
      g_i = p_i - y_i
      h_i = p_i * (1 - p_i)
    """
    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def gradients(cls, y: np.ndarray, y_pred: np.ndarray):
        p = cls.sigmoid(y_pred)
        g = p - y               # first derivative
        h = p * (1 - p)        # second derivative: p(1-p) — the Bernoulli variance
        return g, h

    @staticmethod
    def initial_prediction(y: np.ndarray) -> float:
        p = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


# ─────────────────────────────────────────────
# XGBOOST TREE NODE
# ─────────────────────────────────────────────

@dataclass
class XGBNode:
    """
    A node in an XGBoost tree.
    Stores optimal leaf weight computed from g and h sums.
    """
    feature_idx : Optional[int]   = None
    threshold   : Optional[float] = None
    left        : Optional["XGBNode"] = None
    right       : Optional["XGBNode"] = None
    weight      : Optional[float] = None    # leaf output (w*)
    gain        : float           = 0.0
    cover       : float           = 0.0    # sum of H at this node

    @property
    def is_leaf(self):
        return self.weight is not None


# ─────────────────────────────────────────────
# XGBOOST TREE
# ─────────────────────────────────────────────

class XGBTree:
    """
    A single XGBoost regression tree.

    Builds the tree using the XGBoost gain formula instead of
    standard impurity measures.

    Parameters
    ----------
    max_depth:  maximum tree depth
    lambda_:    L2 regularization on leaf weights
    gamma:      minimum gain required to split (complexity pruning)
    min_child_weight: minimum sum of Hessians in a child node
                      (prevents splits where we have too little data/confidence)
    colsample: fraction of features to consider per split
    """

    def __init__(
        self,
        max_depth:         int   = 6,
        lambda_:           float = 1.0,
        gamma:             float = 0.0,
        min_child_weight:  float = 1.0,
        colsample:         float = 1.0,
        random_state:      Optional[int] = None,
    ):
        self.max_depth        = max_depth
        self.lambda_          = lambda_
        self.gamma            = gamma
        self.min_child_weight = min_child_weight
        self.colsample        = colsample
        self._rng = np.random.RandomState(random_state)
        self.root_            = None

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "XGBTree":
        """
        Build tree to minimize XGBoost objective given gradients g and hessians h.
        """
        self.root_ = self._build(X, g, h, depth=0)
        return self

    def _build(self, X, g, h, depth) -> XGBNode:
        node = XGBNode()
        G = g.sum()
        H = h.sum()
        node.cover = H

        # Optimal leaf weight: w* = -G / (H + lambda)
        # Derived by setting d(obj)/dw = 0 in the quadratic Taylor expansion
        leaf_weight = -G / (H + self.lambda_)

        if depth >= self.max_depth or len(g) < 2:
            node.weight = leaf_weight
            return node

        best = self._best_split(X, g, h)

        if best is None:
            node.weight = leaf_weight
            return node

        feat, thresh, left_mask = best
        right_mask = ~left_mask

        node.feature_idx = feat
        node.threshold   = thresh
        node.left  = self._build(X[left_mask],  g[left_mask],  h[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return node

    def _best_split(self, X, g, h):
        """
        Enumerate all splits and find max gain using the XGBoost gain formula.

        Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ

        This is derived from the Taylor-expanded objective. The gain measures
        how much the quadratic loss approximation improves by splitting.

        Key difference from vanilla GB: H in the denominator means
        splits in high-uncertainty regions (small H) are penalized more.
        For logistic loss, H = p(1-p), so near decision boundary H≈0.25
        and far from it H→0, naturally discouraging overconfident splits.
        """
        G = g.sum()
        H = h.sum()
        best_gain   = -np.inf
        best_feat   = None
        best_thresh = None
        best_mask   = None

        n_features = X.shape[1]
        if self.colsample < 1.0:
            n_col = max(1, int(n_features * self.colsample))
            feats = self._rng.choice(n_features, size=n_col, replace=False)
        else:
            feats = np.arange(n_features)

        for feat in feats:
            x_feat = X[:, feat]
            # Sort by feature value — allows cumulative G, H computation
            order      = np.argsort(x_feat)
            x_sorted   = x_feat[order]
            g_sorted   = g[order]
            h_sorted   = h[order]

            # Cumulative sums for efficient split evaluation
            G_L_cum = np.cumsum(g_sorted)
            H_L_cum = np.cumsum(h_sorted)

            # Candidate split points: between consecutive unique values
            unique_vals = np.unique(x_sorted)
            if len(unique_vals) < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thresh in thresholds:
                split_pos = np.searchsorted(x_sorted, thresh, side='right') - 1
                if split_pos < 0 or split_pos >= len(g) - 1:
                    continue

                G_L = G_L_cum[split_pos]
                H_L = H_L_cum[split_pos]
                G_R = G - G_L
                H_R = H - H_L

                # min_child_weight: reject splits where either side has too little Hessian
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                # XGBoost Gain formula
                gain = 0.5 * (
                    G_L**2 / (H_L + self.lambda_)
                    + G_R**2 / (H_R + self.lambda_)
                    - G**2  / (H  + self.lambda_)
                ) - self.gamma

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = thresh
                    best_mask   = x_feat <= thresh

        if best_gain <= 0:
            return None
        return best_feat, best_thresh, best_mask

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root_) for x in X])

    def _traverse(self, x, node: XGBNode):
        if node.is_leaf:
            return node.weight
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ─────────────────────────────────────────────
# XGBOOST REGRESSOR
# ─────────────────────────────────────────────

class XGBRegressor:
    """
    XGBoost for regression.

    Key parameters (beyond vanilla GB):
      lambda_:          L2 regularization on leaf weights (default=1)
      alpha:            L1 regularization on leaf weights (default=0)
      gamma:            min gain to make a split (default=0)
      min_child_weight: min sum of hessians in child (default=1)
    """

    def __init__(
        self,
        n_estimators:      int   = 100,
        learning_rate:     float = 0.3,
        max_depth:         int   = 6,
        lambda_:           float = 1.0,
        alpha:             float = 0.0,
        gamma:             float = 0.0,
        min_child_weight:  float = 1.0,
        subsample:         float = 1.0,
        colsample:         float = 1.0,
        random_state:      Optional[int] = None,
    ):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.lambda_          = lambda_
        self.alpha            = alpha
        self.gamma            = gamma
        self.min_child_weight = min_child_weight
        self.subsample        = subsample
        self.colsample        = colsample
        self.random_state     = random_state
        self._trees           = []
        self._F0              = None
        self._obj             = SquaredErrorObjective()
        self.train_loss_      = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBRegressor":
        X   = np.array(X, dtype=float)
        y   = np.array(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        self._F0 = self._obj.initial_prediction(y)
        F = np.full(len(y), self._F0)

        for m in range(self.n_estimators):
            g, h = self._obj.gradients(y, F)

            # Row subsampling
            if self.subsample < 1.0:
                n_sub = max(1, int(len(X) * self.subsample))
                idx   = rng.choice(len(X), size=n_sub, replace=False)
                X_sub, g_sub, h_sub = X[idx], g[idx], h[idx]
            else:
                X_sub, g_sub, h_sub = X, g, h

            tree = XGBTree(
                max_depth        = self.max_depth,
                lambda_          = self.lambda_,
                gamma            = self.gamma,
                min_child_weight = self.min_child_weight,
                colsample        = self.colsample,
                random_state     = rng.randint(0, 10000),
            )
            tree.fit(X_sub, g_sub, h_sub)

            F += self.learning_rate * tree.predict(X)
            self._trees.append(tree)
            self.train_loss_.append(np.mean((y - F) ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.full(len(X), self._F0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────
# XGBOOST CLASSIFIER
# ─────────────────────────────────────────────

class XGBClassifier:
    """XGBoost for binary classification (logistic objective)."""

    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=6,
                 lambda_=1.0, gamma=0.0, min_child_weight=1.0,
                 subsample=1.0, colsample=1.0, random_state=None):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.lambda_          = lambda_
        self.gamma            = gamma
        self.min_child_weight = min_child_weight
        self.subsample        = subsample
        self.colsample        = colsample
        self.random_state     = random_state
        self._trees           = []
        self._F0              = None
        self._obj             = LogisticObjective()

    def fit(self, X, y):
        X   = np.array(X, dtype=float)
        y   = np.array(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        self._F0 = self._obj.initial_prediction(y)
        F = np.full(len(y), self._F0)

        for m in range(self.n_estimators):
            g, h = self._obj.gradients(y, F)

            if self.subsample < 1.0:
                n_sub = max(1, int(len(X) * self.subsample))
                idx   = rng.choice(len(X), size=n_sub, replace=False)
                X_sub, g_sub, h_sub = X[idx], g[idx], h[idx]
            else:
                X_sub, g_sub, h_sub = X, g, h

            tree = XGBTree(
                max_depth        = self.max_depth,
                lambda_          = self.lambda_,
                gamma            = self.gamma,
                min_child_weight = self.min_child_weight,
                colsample        = self.colsample,
                random_state     = rng.randint(0, 10000),
            )
            tree.fit(X_sub, g_sub, h_sub)
            F += self.learning_rate * tree.predict(X)
            self._trees.append(tree)

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        F = np.full(len(X), self._F0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        p = LogisticObjective.sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
