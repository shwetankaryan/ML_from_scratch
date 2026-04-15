# ML From Scratch

**KNN, Decision Trees, Gradient Boosting, XGBoost, and CatBoost — implemented in pure NumPy, validated against sklearn.**

Each implementation is annotated with *why* the algorithm works, not just *what* it does. Every scratch implementation is validated against sklearn to confirm correctness.

---

## Algorithms

| Algorithm | File | Key concepts |
|---|---|---|
| KNN (clf + reg) | `src/knn.py` | Lazy learning, vectorized distances, weighted voting |
| Decision Tree (CART) | `src/decision_tree.py` | Gini/entropy/MSE splitting, greedy recursion, feature importance |
| Gradient Boosting | `src/gradient_boosting.py` | First-order gradients, function space gradient descent, staged prediction |
| XGBoost | `src/xgboost_scratch.py` | Second-order gradients, regularized gain formula, optimal leaf weights |
| CatBoost | `src/catboost_scratch.py` | Ordered boosting, ordered target encoding, target leakage prevention |

---

## The Boosting Family — What Each Paper Added

```
AdaBoost (1997)       — reweight misclassified samples
       ↓
Gradient Boosting     — generalize to any differentiable loss via gradient descent
(Friedman, 2001)      — fit trees to negative gradient (pseudo-residuals)
       ↓
XGBoost               — add second-order gradients (Hessian)
(Chen & Guestrin,     — regularize the tree structure in the objective
2016)                 — optimal leaf weights in closed form
       ↓
CatBoost              — ordered boosting (eliminates target leakage)
(Prokhorenkova et al, — ordered target statistics (safe categorical encoding)
2018)
```

---

## Key Equations

**Gradient Boosting update:**
```
F_m(x) = F_{m-1}(x) + lr × h_m(x)
where h_m is a tree fit to: r_i = -∂L/∂F(x_i)
```

**XGBoost optimal leaf weight:**
```
w* = -ΣG / (ΣH + λ)
where G = Σg_i (sum of gradients), H = Σh_i (sum of hessians)
```

**XGBoost split gain:**
```
Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
```

**CatBoost ordered target encoding:**
```
encode_i(c) = (Σ y_j [j<i, cat_j=c] + prior_weight × global_mean)
              / (count(j<i, cat_j=c) + prior_weight)
```

---

## Setup

```bash
pip install numpy scikit-learn pytest matplotlib
pytest tests/ -v
```

---

## Notebooks

Each notebook contains: intuition → math → code → sklearn comparison → where it breaks down.

| Notebook | What you'll see |
|---|---|
| `01_knn.ipynb` | Decision boundary visualization, curse of dimensionality, k selection |
| `02_decision_tree.ipynb` | Tree visualization, impurity plots, feature importance |
| `03_gradient_boosting.ipynb` | Staged prediction, learning rate vs n_estimators trade-off |
| `04_xgboost.ipynb` | G/H plots, gain formula vs impurity, regularization effect |
| `05_catboost.ipynb` | Target leakage demo, ordered encoding vs standard |
