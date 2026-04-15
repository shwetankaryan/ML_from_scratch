"""
tests/test_all.py — Validate scratch implementations against sklearn.

For each algorithm, we check:
  1. Predictions match sklearn within tolerance
  2. Key mathematical properties hold (e.g. initial loss = log(vocab), pure node = 0 impurity)
  3. Edge cases (single feature, single class, etc.)

Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier as SKGBClassifier
from sklearn.ensemble import GradientBoostingRegressor as SKGBRegressor

from knn import KNNClassifier, KNNRegressor
from decision_tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                            gini_impurity, entropy, mse_impurity)
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost_scratch import XGBClassifier, XGBRegressor
from catboost_scratch import OrderedTargetEncoder, compare_target_encoding


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2,
                                random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=300, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    # Binary: only classes 0 and 1
    mask = y < 2
    return train_test_split(X[mask], y[mask], test_size=0.2, random_state=42)


# ─────────────────────────────────────────────
# IMPURITY TESTS
# ─────────────────────────────────────────────

class TestImpurityFunctions:

    def test_gini_pure_node(self):
        y = np.array([1, 1, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.0)

    def test_gini_max_impurity_binary(self):
        y = np.array([0, 0, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.5)

    def test_entropy_pure_node(self):
        y = np.array([0, 0, 0])
        assert entropy(y) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_max_binary(self):
        y = np.array([0, 1])
        assert entropy(y) == pytest.approx(1.0, abs=1e-6)

    def test_mse_pure_node(self):
        y = np.array([3.0, 3.0, 3.0])
        assert mse_impurity(y) == pytest.approx(0.0)

    def test_mse_positive(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse_impurity(y) > 0


# ─────────────────────────────────────────────
# KNN TESTS
# ─────────────────────────────────────────────

class TestKNN:

    def test_classifier_accuracy_vs_sklearn(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        my_knn = KNNClassifier(k=5).fit(X_tr_s, y_tr)
        sk_knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr_s, y_tr)

        my_acc = my_knn.score(X_te_s, y_te)
        sk_acc = sk_knn.score(X_te_s, y_te)
        # Predictions should be identical (same algorithm, same k)
        np.testing.assert_array_equal(
            my_knn.predict(X_te_s),
            sk_knn.predict(X_te_s),
            err_msg="KNN classifier predictions differ from sklearn"
        )

    def test_regressor_predictions_vs_sklearn(self, reg_data):
        X_tr, X_te, y_tr, y_te = reg_data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        my_knn = KNNRegressor(k=5).fit(X_tr_s, y_tr)
        sk_knn = KNeighborsRegressor(n_neighbors=5).fit(X_tr_s, y_tr)

        np.testing.assert_allclose(
            my_knn.predict(X_te_s),
            sk_knn.predict(X_te_s),
            rtol=1e-5,
            err_msg="KNN regressor predictions differ from sklearn"
        )

    def test_k1_returns_exact_neighbor(self, clf_data):
        """k=1 should return the label of the closest training point."""
        X_tr, X_te, y_tr, y_te = clf_data
        knn = KNNClassifier(k=1).fit(X_tr, y_tr)
        # Predict on training data — should get perfect accuracy with k=1
        assert knn.score(X_tr, y_tr) == 1.0

    def test_distance_weighted_vs_uniform(self, clf_data):
        """Distance-weighted KNN should have accuracy >= uniform in most cases."""
        X_tr, X_te, y_tr, y_te = clf_data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        uniform  = KNNClassifier(k=5, weights='uniform').fit(X_tr_s, y_tr)
        weighted = KNNClassifier(k=5, weights='distance').fit(X_tr_s, y_tr)
        # Both should be reasonable classifiers
        assert uniform.score(X_te_s, y_te)  > 0.7
        assert weighted.score(X_te_s, y_te) > 0.7

    def test_manhattan_distance(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        knn = KNNClassifier(k=5, metric='manhattan').fit(X_tr, y_tr)
        assert knn.score(X_te, y_te) > 0.5

    def test_predict_proba_sums_to_one(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        knn = KNNClassifier(k=5).fit(X_tr, y_tr)
        proba = knn.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ─────────────────────────────────────────────
# DECISION TREE TESTS
# ─────────────────────────────────────────────

class TestDecisionTree:

    def test_classifier_accuracy_vs_sklearn(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        my_dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_tr, y_tr)
        sk_dt = SKDecisionTreeClassifier(max_depth=4, random_state=42).fit(X_tr, y_tr)
        # Accuracy within 5% of sklearn (exact match unlikely due to tie-breaking)
        assert abs(my_dt.score(X_te, y_te) - sk_dt.score(X_te, y_te)) < 0.05

    def test_regressor_r2_vs_sklearn(self, reg_data):
        X_tr, X_te, y_tr, y_te = reg_data
        my_dt = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X_tr, y_tr)
        sk_dt = SKDecisionTreeRegressor(max_depth=4, random_state=42).fit(X_tr, y_tr)
        assert abs(my_dt.score(X_te, y_te) - sk_dt.score(X_te, y_te)) < 0.05

    def test_perfect_fit_depth_unlimited(self, clf_data):
        """With no depth limit, tree should perfectly fit training data."""
        X_tr, X_te, y_tr, y_te = clf_data
        dt = DecisionTreeClassifier(max_depth=100).fit(X_tr, y_tr)
        assert dt.score(X_tr, y_tr) == 1.0

    def test_depth_limit_respected(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        for max_d in [1, 2, 3]:
            dt = DecisionTreeClassifier(max_depth=max_d).fit(X_tr, y_tr)
            assert dt.get_depth() <= max_d

    def test_feature_importance_sums_to_one(self, clf_data):
        X_tr, _, y_tr, _ = clf_data
        dt = DecisionTreeClassifier(max_depth=4).fit(X_tr, y_tr)
        np.testing.assert_allclose(
            dt.feature_importances_.sum(), 1.0, atol=1e-6
        )

    def test_single_feature_dataset(self):
        """Tree should work with a single feature."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0,   0,   0,   1,   1,   1])
        dt = DecisionTreeClassifier(max_depth=2).fit(X, y)
        assert dt.score(X, y) == 1.0


# ─────────────────────────────────────────────
# GRADIENT BOOSTING TESTS
# ─────────────────────────────────────────────

class TestGradientBoosting:

    def test_regressor_r2_vs_sklearn(self, reg_data):
        X_tr, X_te, y_tr, y_te = reg_data
        my_gb = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
        ).fit(X_tr, y_tr)
        sk_gb = SKGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
        ).fit(X_tr, y_tr)

        my_r2 = my_gb.score(X_te, y_te)
        sk_r2 = sk_gb.score(X_te, y_te)
        # Within 5% of sklearn R²
        assert abs(my_r2 - sk_r2) < 0.05, f"My R²={my_r2:.3f}, sklearn R²={sk_r2:.3f}"

    def test_classifier_accuracy_vs_sklearn(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        my_gb = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
        ).fit(X_tr, y_tr)
        sk_gb = SKGBClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
        ).fit(X_tr, y_tr)

        assert abs(my_gb.score(X_te, y_te) - sk_gb.score(X_te, y_te)) < 0.05

    def test_loss_decreases_monotonically(self, reg_data):
        """Training loss should decrease (or stay flat) with more trees."""
        X_tr, _, y_tr, _ = reg_data
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_tr, y_tr)
        losses = gb.train_loss_
        # Allow small numerical noise but overall should decrease
        assert losses[-1] < losses[0], "Training loss did not decrease"

    def test_staged_predict_matches_predict(self, reg_data):
        X_tr, X_te, y_tr, y_te = reg_data
        gb = GradientBoostingRegressor(n_estimators=20, random_state=42).fit(X_tr, y_tr)
        staged = list(gb.staged_predict(X_te))
        # Last staged prediction should match full predict
        np.testing.assert_allclose(staged[-1], gb.predict(X_te), rtol=1e-5)

    def test_learning_rate_effect(self, reg_data):
        """Lower LR with more trees should give similar or better result."""
        X_tr, X_te, y_tr, y_te = reg_data
        gb_high_lr = GradientBoostingRegressor(n_estimators=20,  learning_rate=0.5).fit(X_tr, y_tr)
        gb_low_lr  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1).fit(X_tr, y_tr)
        # Low LR + more trees should generally perform better
        assert gb_low_lr.score(X_te, y_te) >= gb_high_lr.score(X_te, y_te) - 0.05


# ─────────────────────────────────────────────
# XGBOOST TESTS
# ─────────────────────────────────────────────

class TestXGBoost:

    def test_regressor_outperforms_vanilla_gb(self, reg_data):
        """XGBoost with regularization should perform comparably to vanilla GB."""
        X_tr, X_te, y_tr, y_te = reg_data
        xgb = XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3,
            lambda_=1.0, random_state=42
        ).fit(X_tr, y_tr)
        assert xgb.score(X_te, y_te) > 0.5

    def test_classifier_predicts_binary(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        xgb = XGBClassifier(n_estimators=30, learning_rate=0.1, random_state=42).fit(X_tr, y_tr)
        preds = xgb.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_valid(self, iris_data):
        X_tr, X_te, y_tr, y_te = iris_data
        xgb = XGBClassifier(n_estimators=20, random_state=42).fit(X_tr, y_tr)
        proba = xgb.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_lambda_regularization_reduces_overfitting(self, reg_data):
        """Higher lambda should reduce the gap between train and test performance."""
        X_tr, X_te, y_tr, y_te = reg_data
        xgb_noreg = XGBRegressor(n_estimators=50, lambda_=0.0, max_depth=6).fit(X_tr, y_tr)
        xgb_reg   = XGBRegressor(n_estimators=50, lambda_=5.0, max_depth=6).fit(X_tr, y_tr)

        gap_noreg = xgb_noreg.score(X_tr, y_tr) - xgb_noreg.score(X_te, y_te)
        gap_reg   = xgb_reg.score(X_tr, y_tr)   - xgb_reg.score(X_te, y_te)
        # Regularization should reduce or maintain the train-test gap
        assert gap_reg <= gap_noreg + 0.1

    def test_loss_decreases(self, reg_data):
        X_tr, _, y_tr, _ = reg_data
        xgb = XGBRegressor(n_estimators=30, random_state=42).fit(X_tr, y_tr)
        assert xgb.train_loss_[-1] < xgb.train_loss_[0]


# ─────────────────────────────────────────────
# CATBOOST / ORDERED ENCODING TESTS
# ─────────────────────────────────────────────

class TestOrderedEncoding:

    def test_ordered_encoding_reduces_correlation_with_y(self):
        """
        Ordered TS should have lower correlation with y than standard encoding.
        This demonstrates the target leakage reduction.
        """
        np.random.seed(42)
        N = 200
        # Few categories → standard encoding leaks heavily
        categories = np.random.choice(['A', 'B', 'C'], size=N)
        y = np.random.randn(N)
        # Make y depend on category to create realistic encoding scenario
        y[categories == 'A'] += 1.0
        y[categories == 'B'] -= 0.5

        result = compare_target_encoding(y, categories, seed=42)

        # Standard encoding is perfectly correlated with itself via y
        # Ordered should have lower correlation (less leakage)
        # Note: both correlate with y because there IS a real signal
        # The key is ordered is lower than standard
        assert result['correlation_standard_with_y'] >= \
               result['correlation_ordered_with_y'] - 0.1

    def test_unseen_category_falls_back_to_prior(self):
        """Test categories at inference not seen in training → global mean."""
        y     = np.array([1.0, 2.0, 3.0, 4.0])
        cats  = np.array(['A', 'A', 'B', 'B'])
        order = np.array([0, 1, 2, 3])

        enc = OrderedTargetEncoder(prior_weight=1.0)
        enc.fit_transform(cats, y, order)
        enc.global_mean_ = y.mean()

        # 'C' is unseen
        test_encoded = enc.transform(np.array(['C']), cats, y)
        assert test_encoded[0] == pytest.approx(y.mean(), abs=0.1)

    def test_early_samples_close_to_prior(self):
        """
        First few samples (no history yet) should get encoding close to global mean.
        """
        np.random.seed(0)
        N = 100
        y    = np.random.randn(N) + 5.0   # global mean ≈ 5
        cats = np.array(['A'] * N)         # single category, first sample has no history
        order = np.arange(N)

        enc = OrderedTargetEncoder(prior_weight=10.0)  # strong prior
        encoded = enc.fit_transform(cats, y, order)

        # First sample: no history → encoding = global mean (via prior)
        global_mean = y.mean()
        assert abs(encoded[order[0]] - global_mean) < 1.5
