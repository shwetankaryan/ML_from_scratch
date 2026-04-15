"""
catboost_scratch.py — CatBoost core ideas from scratch (numpy only).

CatBoost (Prokhorenkova et al., 2018) introduced two key innovations over
XGBoost and LightGBM:

  1. ORDERED BOOSTING (prevents target leakage)
  2. ORDERED TARGET STATISTICS (better categorical encoding)

──────────────────────────────────────────────────────────────────────────

PROBLEM: TARGET LEAKAGE IN STANDARD GRADIENT BOOSTING
═══════════════════════════════════════════════════════

In standard GB, when building tree m:
  - We compute residuals on the FULL training set
  - We then fit a tree to those residuals using the FULL training set
  - But the previous trees (F_{m-1}) were already fit using the same data

This creates a subtle data leakage: the residuals for sample i were
influenced by sample i itself (when training earlier trees).

Result: the model overfits on training data in a way that's hard to detect.
Symptom: training and test performance diverge more than expected.

──────────────────────────────────────────────────────────────────────────

SOLUTION 1: ORDERED BOOSTING
═════════════════════════════

CatBoost maintains M separate models F^1, F^2, ..., F^N (one per sample).
When computing the residual for sample i, it uses a model that was trained
on samples {1, ..., i-1} only — i.e., sample i has never been seen by
the model used to compute its own residual.

This completely eliminates target leakage in residual computation.

In practice (and in this implementation): we shuffle training samples,
then for sample i, we compute its residual using a model trained on [0..i-1].

──────────────────────────────────────────────────────────────────────────

SOLUTION 2: ORDERED TARGET STATISTICS (for categoricals)
═══════════════════════════════════════════════════════════

Standard target encoding:
  encode(category) = mean(target | category)

Problem: for sample i with category c, if i is the only sample with c,
  encode = y_i exactly → the model can perfectly "predict" y_i by encoding!
  This is extreme target leakage.

CatBoost's solution (Ordered TS):
  For sample i, only use samples {1..i-1} to compute the encoding:
  encode_i(c) = (sum(y_j | j < i, category_j == c) + prior * prior_weight)
                / (count(j < i, category_j == c) + prior_weight)

  For early samples (few examples of category c seen so far), this falls back
  to the prior (global mean), smoothly interpolating as more data is seen.
"""

import numpy as np
from typing_extensions import Optional, Dict, List
from decision_tree import DecisionTreeRegressor


# ─────────────────────────────────────────────
# ORDERED TARGET STATISTICS
# ─────────────────────────────────────────────

class OrderedTargetEncoder:
    """
    CatBoost-style ordered target encoding for categorical features.

    For each sample i (in a fixed random order), the encoding of its
    category is computed using only samples that came BEFORE i in that order.

    This prevents target leakage that plagues standard mean encoding.

    Parameters
    ----------
    prior_weight : float
        Weight given to the global prior (global mean of y).
        Higher = more regularization, falls back to global mean for rare categories.
    """

    def __init__(self, prior_weight: float = 1.0):
        self.prior_weight = prior_weight
        self.global_mean_ = None

    def fit_transform(
        self,
        categories: np.ndarray,   # (N,) categorical feature column
        y: np.ndarray,            # (N,) targets
        order: np.ndarray,        # (N,) permutation indices
    ) -> np.ndarray:
        """
        Compute ordered target statistics for training.

        Returns encoded values for each sample using only
        the "past" samples in the given permutation order.
        """
        N = len(categories)
        self.global_mean_ = y.mean()

        encoded = np.zeros(N)
        # Running sums per category: cat → (sum_y, count)
        cat_sum   = {}
        cat_count = {}

        for pos in range(N):
            i   = order[pos]
            cat = categories[i]

            # Encoding for sample i: use what's seen so far (before position pos)
            cat_s = cat_sum.get(cat, 0.0)
            cat_c = cat_count.get(cat, 0)

            encoded[i] = (cat_s + self.prior_weight * self.global_mean_) / \
                         (cat_c + self.prior_weight)

            # Update running stats with sample i's target
            cat_sum[cat]   = cat_s + y[i]
            cat_count[cat] = cat_c + 1

        return encoded

    def transform(
        self,
        categories: np.ndarray,
        full_categories: np.ndarray,
        full_y: np.ndarray,
    ) -> np.ndarray:
        """
        Encode test data using full training statistics.
        (No ordering needed at inference — use all training data.)
        """
        cat_stats = {}
        for cat, yi in zip(full_categories, full_y):
            if cat not in cat_stats:
                cat_stats[cat] = [0.0, 0]
            cat_stats[cat][0] += yi
            cat_stats[cat][1] += 1

        encoded = np.zeros(len(categories))
        for i, cat in enumerate(categories):
            if cat in cat_stats:
                s, c = cat_stats[cat]
                encoded[i] = (s + self.prior_weight * self.global_mean_) / \
                             (c + self.prior_weight)
            else:
                # Unseen category → fall back to global mean
                encoded[i] = self.global_mean_
        return encoded


# ─────────────────────────────────────────────
# ORDERED BOOSTING
# ─────────────────────────────────────────────

class OrderedBoostingRegressor:
    """
    CatBoost-style Ordered Boosting for regression.

    Implements the core ordered boosting algorithm:
      For each tree m:
        1. Sample a random permutation π of training indices
        2. For each sample i, compute residual using model trained on π[0..i-1]
        3. Fit a tree to these "clean" residuals

    In this simplified implementation:
      - We maintain one growing model for computing ordered residuals
      - We use subsampling to approximate the per-sample model

    Notes on the approximation:
      Full CatBoost maintains O(n_estimators) separate models for exact
      ordered boosting, which is memory-intensive. We use a practical
      approximation where we shuffle and compute residuals on held-out portions.
    """

    def __init__(
        self,
        n_estimators:  int   = 100,
        learning_rate: float = 0.1,
        max_depth:     int   = 4,
        cat_features:  Optional[List[int]] = None,   # indices of categorical columns
        prior_weight:  float = 1.0,
        random_state:  Optional[int] = None,
    ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.cat_features  = cat_features or []
        self.prior_weight  = prior_weight
        self.random_state  = random_state
        self._trees        = []
        self._F0           = None
        self._encoders     = {}
        self._train_cats   = {}   # store training categoricals for test encoding
        self._train_y      = None
        self.train_loss_   = []

    def _encode_categoricals(self, X: np.ndarray, y: np.ndarray,
                              order: np.ndarray, is_train: bool) -> np.ndarray:
        """
        Apply ordered target encoding to all categorical columns.
        Numeric columns pass through unchanged.
        """
        X_encoded = X.astype(float).copy()

        for feat_idx in self.cat_features:
            if is_train:
                enc = OrderedTargetEncoder(self.prior_weight)
                X_encoded[:, feat_idx] = enc.fit_transform(
                    X[:, feat_idx].astype(str), y, order
                )
                self._encoders[feat_idx]   = enc
                self._train_cats[feat_idx] = X[:, feat_idx].astype(str)
            else:
                enc = self._encoders[feat_idx]
                X_encoded[:, feat_idx] = enc.transform(
                    X[:, feat_idx].astype(str),
                    self._train_cats[feat_idx],
                    self._train_y,
                )
        return X_encoded

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrderedBoostingRegressor":
        X   = np.array(X)
        y   = np.array(y, dtype=float)
        N   = len(y)
        rng = np.random.RandomState(self.random_state)

        self._train_y = y.copy()
        self._F0 = y.mean()

        # One prediction per training sample (updated incrementally)
        F = np.full(N, self._F0)

        for m in range(self.n_estimators):
            # ── Step 1: Sample random permutation ──────────────────
            order = rng.permutation(N)

            # ── Step 2: Ordered target encoding of categoricals ─────
            X_enc = self._encode_categoricals(X, y, order, is_train=True)

            # ── Step 3: Compute ordered residuals ───────────────────
            # Approximation: split data into two halves.
            # Half A: use F computed from half B as the "past" model.
            # This gives unbiased residuals for half A.
            mid         = N // 2
            first_half  = order[:mid]
            second_half = order[mid:]

            residuals = np.zeros(N)

            # For first half: residual uses model state from second half "history"
            # (approximation — true CatBoost uses a per-sample model)
            residuals[first_half]  = y[first_half]  - F[second_half].mean()
            residuals[second_half] = y[second_half] - F[first_half].mean()

            # More accurate: actual ordered residuals using a growing model
            # (slower but truer to the paper)
            F_ordered = np.full(N, self._F0)
            for pos in range(1, N):
                i = order[pos]
                # Sample i's residual uses model fit on order[0..pos-1]
                residuals[i] = y[i] - F_ordered[order[:pos]].mean()
                # Update F_ordered for sample i with previous tree predictions
                if m > 0:
                    for tree in self._trees:
                        F_ordered[i] += self.learning_rate * tree.predict(X_enc[[i]])[0]

            # ── Step 4: Fit tree to ordered residuals ───────────────
            tree = DecisionTreeRegressor(
                max_depth = self.max_depth,
                random_state = rng.randint(0, 10000),
            )
            tree.fit(X_enc, residuals)

            # ── Step 5: Update predictions ──────────────────────────
            F += self.learning_rate * tree.predict(X_enc)
            self._trees.append(tree)

            loss = np.mean((y - F) ** 2)
            self.train_loss_.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X     = np.array(X)
        order = np.arange(len(X))   # fixed order for test (no randomness needed)
        X_enc = self._encode_categoricals(X, self._train_y, order, is_train=False)

        F = np.full(len(X), self._F0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X_enc)
        return F

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────
# STANDALONE COMPARISON UTILITY
# ─────────────────────────────────────────────

def compare_target_encoding(y: np.ndarray, categories: np.ndarray, seed: int = 42):
    """
    Side-by-side comparison of standard mean encoding vs ordered TS.
    Shows the leakage problem and how CatBoost fixes it.
    """
    N   = len(y)
    rng = np.random.RandomState(seed)
    order = rng.permutation(N)

    # Standard mean encoding (leaks)
    standard = {}
    for cat, yi in zip(categories, y):
        standard.setdefault(cat, []).append(yi)
    standard_enc = np.array([np.mean(standard[c]) for c in categories])

    # Ordered target encoding (CatBoost)
    enc = OrderedTargetEncoder(prior_weight=1.0)
    ordered_enc = enc.fit_transform(categories, y, order)

    return {
        'standard_encoding': standard_enc,
        'ordered_encoding':  ordered_enc,
        'correlation_standard_with_y': np.corrcoef(standard_enc, y)[0, 1],
        'correlation_ordered_with_y':  np.corrcoef(ordered_enc,  y)[0, 1],
    }
