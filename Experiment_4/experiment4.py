"""
Experiment 4: Binary Classification using Linear and Kernel-Based Models
ICS1512 Machine Learning Algorithms Laboratory

Dataset: Spambase (Kaggle - somesh24/spambase)

Steps:
    1. Load and preprocess data
    2. Exploratory Data Analysis
    3. Train baseline Logistic Regression, tune with GridSearchCV
    4. Train SVM with 4 kernels (Linear, Polynomial, RBF, Sigmoid), tune with GridSearchCV
    5. Evaluate all models with standard metrics
    6. 5-Fold Cross-Validation
    7. Save all required plots and comparison tables
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eda_toolkit import run_eda

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "outputs")
DATADIR = os.path.join(HERE, "data")
os.makedirs(OUTDIR, exist_ok=True)


def save_plot(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


def save_table(df, filename):
    path = os.path.join(OUTDIR, filename)
    df.to_csv(path, index=False)
    print("saved:", path)


# 1. load the data
print("loading data")
df = pd.read_csv(os.path.join(DATADIR, "spambase_csv.csv"))
print("shape:", df.shape)
print("missing values:", df.isna().sum().sum())

X = df.drop(columns=["class"])
y = df["class"]

# 2. exploratory data analysis
print("\nrunning EDA")

fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts().sort_index()
ax.bar(["ham (0)", "spam (1)"], counts.values, color=["seagreen", "indianred"])
for i, v in enumerate(counts.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set_title("Class distribution")
save_plot(fig, "01_class_distribution.png")

corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
top_features = corr_with_target.head(10).index.tolist()
run_eda(df[top_features + ["class"]], target="class",
        outdir=os.path.join(OUTDIR, "eda"), name="spambase")

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(df.corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Correlation heatmap - all features + class")
fig.colorbar(im, ax=ax, fraction=0.03)
save_plot(fig, "02_correlation_heatmap_full.png")

# 3. split and standardize
print("\npreprocessing")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"train size: {len(X_train)}, test size: {len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_te)
    predict_time = time.perf_counter() - t0

    try:
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
    except Exception:
        proba, auc = None, np.nan

    return {
        "model": model, "pred": pred, "proba": proba,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": auc,
        "train_time": train_time, "predict_time": predict_time,
    }


# 4. Logistic Regression: baseline, then tuned with GridSearchCV
print("\ntraining Logistic Regression")
baseline_lr = evaluate(LogisticRegression(max_iter=5000), X_train_s, y_train, X_test_s, y_test)
print("baseline Logistic Regression accuracy:", baseline_lr["accuracy"])

lr_param_grid = [
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear"]},
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["saga"]},
]
t0 = time.perf_counter()
lr_grid = GridSearchCV(LogisticRegression(max_iter=5000), lr_param_grid, cv=5,
                        scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train_s, y_train)
lr_grid_time = time.perf_counter() - t0
print("Logistic Regression best params:", lr_grid.best_params_, " best CV acc:", lr_grid.best_score_)

lr_best = evaluate(LogisticRegression(max_iter=5000, **lr_grid.best_params_),
                    X_train_s, y_train, X_test_s, y_test)
print("tuned Logistic Regression test accuracy:", lr_best["accuracy"])

lr_perf_table = pd.DataFrame([{
    "Metric": "Accuracy", "Value": round(lr_best["accuracy"], 4)}, {
    "Metric": "Precision", "Value": round(lr_best["precision"], 4)}, {
    "Metric": "Recall", "Value": round(lr_best["recall"], 4)}, {
    "Metric": "F1 Score", "Value": round(lr_best["f1"], 4)}, {
    "Metric": "Training Time (s)", "Value": round(lr_best["train_time"], 5)},
])
save_table(lr_perf_table, "table_logistic_regression_performance.csv")

# 5. SVM across 4 kernels, tuned with GridSearchCV
print("\ntraining SVM (linear, polynomial, RBF, sigmoid kernels)")
svm_param_grid = [
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    {"kernel": ["sigmoid"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    {"kernel": ["poly"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"], "degree": [2, 3]},
]
t0 = time.perf_counter()
svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train_s, y_train)
svm_grid_time = time.perf_counter() - t0
print("SVM best params:", svm_grid.best_params_, " best CV acc:", svm_grid.best_score_)

tuning_table = pd.DataFrame([
    {"Model": "Logistic Regression", "Search Method": "Grid Search",
     "Best Parameters": str(lr_grid.best_params_), "Best CV Accuracy": round(lr_grid.best_score_, 4)},
    {"Model": "SVM", "Search Method": "Grid Search",
     "Best Parameters": str(svm_grid.best_params_), "Best CV Accuracy": round(svm_grid.best_score_, 4)},
])
save_table(tuning_table, "table_hyperparameter_tuning_results.csv")

# grab the best params per individual kernel from the grid search results,
# so we can compare kernels on their own best settings rather than one global winner
cv_results = pd.DataFrame(svm_grid.cv_results_)
kernel_best_params = {}
for kernel in ["linear", "rbf", "sigmoid", "poly"]:
    sub = cv_results[cv_results["param_kernel"] == kernel]
    best_row = sub.loc[sub["mean_test_score"].idxmax()]
    kernel_best_params[kernel] = {k: v for k, v in best_row["params"].items()}

print("best parameters per kernel:", kernel_best_params)

svm_kernel_results = {}
for kernel, params in kernel_best_params.items():
    model = SVC(probability=True, **params)
    svm_kernel_results[kernel] = evaluate(model, X_train_s, y_train, X_test_s, y_test)
    print(f"{kernel}: accuracy={svm_kernel_results[kernel]['accuracy']:.4f}  "
          f"train_time={svm_kernel_results[kernel]['train_time']:.3f}s")

svm_kernel_table = pd.DataFrame({
    kernel.capitalize(): {"Accuracy": round(r["accuracy"], 4), "F1 Score": round(r["f1"], 4),
                           "Training Time (s)": round(r["train_time"], 4)}
    for kernel, r in svm_kernel_results.items()
}).T.reset_index().rename(columns={"index": "Kernel"})
print("\nSVM kernel-wise performance:\n", svm_kernel_table)
save_table(svm_kernel_table, "table_svm_kernel_performance.csv")

best_kernel = max(svm_kernel_results, key=lambda k: svm_kernel_results[k]["accuracy"])
best_svm = svm_kernel_results[best_kernel]
print("best SVM kernel:", best_kernel)

# 6. confusion matrices, ROC and precision-recall curves
print("\nplotting confusion matrices, ROC and precision-recall curves")
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_best["pred"]), display_labels=["ham", "spam"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Logistic Regression (acc={lr_best['accuracy']:.3f})")
for ax, (kernel, r) in zip(axes[1:], svm_kernel_results.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, r["pred"]), display_labels=["ham", "spam"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"SVM-{kernel} (acc={r['accuracy']:.3f})")
fig.suptitle("Confusion matrices - all models")
save_plot(fig, "03_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(7, 7))
fpr, tpr, _ = roc_curve(y_test, lr_best["proba"])
ax.plot(fpr, tpr, label=f"Logistic Regression (AUC={lr_best['roc_auc']:.3f})", linewidth=2)
for kernel, r in svm_kernel_results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, label=f"SVM-{kernel} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves - all models")
ax.legend()
save_plot(fig, "04_roc_curves.png")

fig, ax = plt.subplots(figsize=(7, 7))
prec, rec, _ = precision_recall_curve(y_test, lr_best["proba"])
ax.plot(rec, prec, label="Logistic Regression", linewidth=2)
for kernel, r in svm_kernel_results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    ax.plot(rec, prec, label=f"SVM-{kernel}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves - all models")
ax.legend()
save_plot(fig, "05_precision_recall_curves.png")

# 7. what do these boundaries actually look like? project down to 2D with PCA
print("\nplotting decision boundaries (PCA 2D)")
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_s)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 200),
                      np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 200))
for ax, kernel in zip(axes, ["linear", "rbf", "sigmoid", "poly"]):
    clf2d = SVC(kernel=kernel, C=1, gamma="scale").fit(X_train_2d, y_train)
    Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="coolwarm", s=5, edgecolor="k", linewidth=0.2)
    ax.set_title(f"SVM kernel = {kernel}")
fig.suptitle("SVM decision boundaries on first 2 PCA components (illustrative)")
save_plot(fig, "06_svm_decision_boundaries_pca.png")

# 8. 5-fold cross validation, Logistic Regression vs the best SVM kernel
print("\nrunning 5-fold cross validation")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(LogisticRegression(max_iter=5000, **lr_grid.best_params_),
                                X_train_s, y_train, cv=skf)
svm_cv_scores = cross_val_score(SVC(**kernel_best_params[best_kernel]), X_train_s, y_train, cv=skf)

cv_table = pd.DataFrame({
    "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"],
    "Logistic Regression": list(np.round(lr_cv_scores, 4)) + [round(lr_cv_scores.mean(), 4)],
    "SVM": list(np.round(svm_cv_scores, 4)) + [round(svm_cv_scores.mean(), 4)],
})
print("\nk-fold cross validation results:\n", cv_table)
save_table(cv_table, "table_kfold_cross_validation.csv")

fig, ax = plt.subplots(figsize=(6, 4))
folds = np.arange(1, 6)
ax.plot(folds, lr_cv_scores, marker="o", label="Logistic Regression")
ax.plot(folds, svm_cv_scores, marker="s", label=f"SVM ({best_kernel})")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_xticks(folds)
ax.set_title("5-Fold Cross-Validation Accuracy")
ax.legend()
save_plot(fig, "07_cross_validation_accuracy.png")

# 9. training time and overall accuracy, side by side
print("\ncomparing training time and accuracy across models")
fig, ax = plt.subplots(figsize=(8, 5))
names = ["Logistic Regression"] + [f"SVM-{k}" for k in svm_kernel_results]
times = [lr_best["train_time"]] + [r["train_time"] for r in svm_kernel_results.values()]
ax.bar(names, times, color=["steelblue"] + ["salmon"] * len(svm_kernel_results))
ax.set_ylabel("seconds"); ax.set_title("Training time comparison")
ax.tick_params(axis="x", rotation=20)
save_plot(fig, "08_training_time_comparison.png")

fig, ax = plt.subplots(figsize=(8, 5))
accs = [lr_best["accuracy"]] + [r["accuracy"] for r in svm_kernel_results.values()]
ax.bar(names, accs, color=["steelblue"] + ["darkorange"] * len(svm_kernel_results))
ax.set_ylabel("Accuracy"); ax.set_title("Classifier comparison - test accuracy")
ax.tick_params(axis="x", rotation=20)
save_plot(fig, "09_classifier_comparison_bar_chart.png")

# 10. quick side-by-side comparative table
comparative_table = pd.DataFrame([
    {"Criterion": "Accuracy", "Logistic Regression": round(lr_best["accuracy"], 4),
     "SVM": round(best_svm["accuracy"], 4)},
    {"Criterion": "Model Complexity", "Logistic Regression": "Low", "SVM": "High"},
    {"Criterion": "Training Time", "Logistic Regression": "Low", "SVM": "High"},
    {"Criterion": "Interpretability", "Logistic Regression": "High", "SVM": "Low"},
])
print("\ncomparative analysis:\n", comparative_table)
save_table(comparative_table, "table_comparative_analysis.csv")

# 11. write up the observations using the numbers computed above
best_overall = "Logistic Regression" if lr_best["accuracy"] >= best_svm["accuracy"] else f"SVM ({best_kernel})"
observations = f"""Experiment 4 - Observations (auto-generated from results)

Best-performing classifier: {best_overall}
   Logistic Regression test accuracy = {lr_best['accuracy']:.4f}
   Best SVM ({best_kernel} kernel) test accuracy = {best_svm['accuracy']:.4f}

Impact of regularization (Logistic Regression):
   GridSearchCV chose {lr_grid.best_params_}. A smaller C means stronger regularization
   (simpler decision boundary, less overfitting); the chosen C={lr_grid.best_params_['C']} and
   penalty={lr_grid.best_params_['penalty']} balances fitting the training data against
   keeping coefficients small.

Kernel behaviour in SVM:
{svm_kernel_table.to_string(index=False)}
   The linear kernel is fastest and already competitive because Spambase is close to linearly
   separable in its 57-dimensional feature space; RBF/poly/sigmoid add non-linear flexibility
   at a higher training-time cost, which only pays off if there is genuine non-linear structure
   left for them to capture (see the 2D PCA decision-boundary plot for a visual comparison).

Bias-variance trade-off:
   Logistic Regression has higher bias / lower variance (a single linear decision boundary),
   while SVM with RBF/poly/sigmoid kernels has lower bias / higher variance (flexible, curved
   boundaries that can overfit if C or gamma is too large) - this is why hyperparameter tuning
   of C and gamma matters so much for SVM performance.
"""
with open(os.path.join(OUTDIR, "observations.txt"), "w") as f:
    f.write(observations)
print(observations)

print("\nExperiment 4 done, everything is in:", OUTDIR)
