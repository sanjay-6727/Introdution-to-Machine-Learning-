"""
Experiment 2: Email Spam/Ham Classification using Naive Bayes and KNN
ICS1512 Machine Learning Algorithms Laboratory

Dataset: Spambase (Kaggle - somesh24/spambase)

Steps:
    1. Load and preprocess data
    2. Exploratory Data Analysis
    3. Train Gaussian, Multinomial, Bernoulli Naive Bayes
    4. Train KNN for several values of k
    5. Tune KNN with GridSearchCV and RandomizedSearchCV
    6. Compare KDTree vs BallTree
    7. 5-Fold Cross Validation
    8. Measure training/prediction time
    9. Save all required plots and comparison tables
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      RandomizedSearchCV, StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
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

feature_cols = [c for c in df.columns if c != "class"]
X = df[feature_cols]
y = df["class"]

# 2. exploratory data analysis
print("\nrunning EDA")

# class distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts().sort_index()
ax.bar(["ham (0)", "spam (1)"], counts.values, color=["seagreen", "indianred"])
for i, v in enumerate(counts.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set_title("Class distribution")
ax.set_ylabel("number of emails")
save_plot(fig, "01_class_distribution.png")

# correlation heatmap, histograms and boxplots, just for the features most
# correlated with spam so the grid stays readable
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
top_features = corr_with_target.head(10).index.tolist()

run_eda(df[top_features + ["class"]], target="class",
        outdir=os.path.join(OUTDIR, "eda"), name="spambase")

# and a full correlation heatmap across every feature, not just the top 10
fig, ax = plt.subplots(figsize=(14, 12))
full_corr = df.corr()
im = ax.imshow(full_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Correlation heatmap - all 57 features + class")
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


def evaluate(model, X_tr, y_tr, X_te, y_te, proba_from=None):
    """Fits a model, times it, and returns the usual classification metrics."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_te)
    predict_time = time.perf_counter() - t0

    proba_model = proba_from if proba_from is not None else model
    try:
        proba = proba_model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
    except Exception:
        proba, auc = None, np.nan

    return {
        "model": model,
        "pred": pred,
        "proba": proba,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": auc,
        "train_time": train_time,
        "predict_time": predict_time,
    }


# 4. Naive Bayes: Gaussian, Multinomial, Bernoulli
print("\ntraining Naive Bayes models")
# GaussianNB wants standardized real-valued features. MultinomialNB and
# BernoulliNB expect non-negative "count-like" input, so they get the raw,
# unscaled word/character frequencies instead.
nb_results = {
    "Gaussian": evaluate(GaussianNB(), X_train_s, y_train, X_test_s, y_test),
    "Multinomial": evaluate(MultinomialNB(), X_train, y_train, X_test, y_test),
    "Bernoulli": evaluate(BernoulliNB(), X_train, y_train, X_test, y_test),
}

nb_table = pd.DataFrame({
    name: {"Accuracy": r["accuracy"], "Precision": r["precision"],
           "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
    for name, r in nb_results.items()
}).round(4)
print("\nNaive Bayes comparison:\n", nb_table)
save_table(nb_table.reset_index().rename(columns={"index": "Metric"}), "table_naive_bayes_comparison.csv")

best_nb_name = max(nb_results, key=lambda k: nb_results[k]["accuracy"])
best_nb = nb_results[best_nb_name]
print("best Naive Bayes variant:", best_nb_name)

# confusion matrix per NB model
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, r) in zip(axes, nb_results.items()):
    cm = confusion_matrix(y_test, r["pred"])
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} NB (acc={r['accuracy']:.3f})")
fig.suptitle("Confusion matrices - Naive Bayes variants")
save_plot(fig, "05a_confusion_matrix_naive_bayes.png")

# ROC and precision-recall curves for the NB variants
fig, ax = plt.subplots(figsize=(6, 6))
for name, r in nb_results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves - Naive Bayes variants")
ax.legend()
save_plot(fig, "06a_roc_curve_naive_bayes.png")

fig, ax = plt.subplots(figsize=(6, 6))
for name, r in nb_results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    ax.plot(rec, prec, label=name)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves - Naive Bayes variants")
ax.legend()
save_plot(fig, "07a_precision_recall_naive_bayes.png")

# 5. KNN for a few values of k
print("\ntrying KNN for a few values of k")
k_values = [1, 3, 5, 7, 9, 11]
knn_rows = []
knn_results = {}
for k in k_values:
    r = evaluate(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, X_test_s, y_test)
    knn_results[k] = r
    knn_rows.append({"k": k, "Accuracy": r["accuracy"], "Precision": r["precision"],
                      "Recall": r["recall"], "F1": r["f1"]})

knn_table = pd.DataFrame(knn_rows).round(4)
print("\nKNN comparison:\n", knn_table)
save_table(knn_table, "table_knn_comparison.csv")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(knn_table["k"], knn_table["Accuracy"], marker="o", color="darkorange")
ax.set_xlabel("k (number of neighbors)")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy vs k (small k -> overfitting, large k -> underfitting)")
save_plot(fig, "08_accuracy_vs_k.png")

best_k = int(knn_table.loc[knn_table["Accuracy"].idxmax(), "k"])
print("best k from the sweep:", best_k)

# 6. GridSearchCV and RandomizedSearchCV for KNN
print("\ntuning KNN with GridSearchCV and RandomizedSearchCV")
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["kd_tree", "ball_tree"],
}

t0 = time.perf_counter()
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_s, y_train)
grid_time = time.perf_counter() - t0
print("GridSearchCV best params:", grid.best_params_, " best CV acc:", grid.best_score_)

t0 = time.perf_counter()
rand_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=15, cv=5,
                                  scoring="accuracy", n_jobs=-1, random_state=42)
rand_search.fit(X_train_s, y_train)
rand_time = time.perf_counter() - t0
print("RandomizedSearchCV best params:", rand_search.best_params_, " best CV acc:", rand_search.best_score_)

search_table = pd.DataFrame({
    "GridSearchCV": {
        "Best k": grid.best_params_["n_neighbors"],
        "Metric": grid.best_params_["metric"],
        "Weights": grid.best_params_["weights"],
        "Algorithm": grid.best_params_["algorithm"],
        "CV Accuracy": round(grid.best_score_, 4),
        "Execution Time (s)": round(grid_time, 3),
    },
    "RandomizedSearchCV": {
        "Best k": rand_search.best_params_["n_neighbors"],
        "Metric": rand_search.best_params_["metric"],
        "Weights": rand_search.best_params_["weights"],
        "Algorithm": rand_search.best_params_["algorithm"],
        "CV Accuracy": round(rand_search.best_score_, 4),
        "Execution Time (s)": round(rand_time, 3),
    },
})
print("\ngrid search vs randomized search:\n", search_table)
save_table(search_table.reset_index().rename(columns={"index": "Parameter"}), "table_grid_vs_randomized_search.csv")

# use the grid search winner as "best KNN" going forward
best_knn_params = grid.best_params_
best_knn = KNeighborsClassifier(**best_knn_params)
best_knn_result = evaluate(best_knn, X_train_s, y_train, X_test_s, y_test)
print("best KNN test accuracy:", best_knn_result["accuracy"])

# heatmap of n_neighbors vs weights, averaged over metric/algorithm
cv_results = pd.DataFrame(grid.cv_results_)
pivot = cv_results.pivot_table(values="mean_test_score", index="param_n_neighbors", columns="param_weights")
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
ax.set_xlabel("weights"); ax.set_ylabel("n_neighbors")
ax.set_title("GridSearchCV mean CV accuracy heatmap")
fig.colorbar(im, ax=ax, fraction=0.046)
save_plot(fig, "13_gridsearch_heatmap.png")

# and the spread of scores RandomizedSearchCV actually sampled
rand_scores = pd.DataFrame(rand_search.cv_results_)["mean_test_score"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(rand_scores, bins=10, color="mediumpurple", edgecolor="white")
ax.axvline(rand_search.best_score_, color="red", linestyle="--", label="best score")
ax.set_xlabel("mean CV accuracy"); ax.set_ylabel("number of parameter combinations")
ax.set_title("RandomizedSearchCV score distribution")
ax.legend()
save_plot(fig, "14_randomizedsearch_score_distribution.png")

# 7. KDTree vs BallTree
print("\ncomparing KDTree and BallTree")
tree_rows = {}
for algo in ["kd_tree", "ball_tree"]:
    model = KNeighborsClassifier(n_neighbors=best_k, algorithm=algo)
    r = evaluate(model, X_train_s, y_train, X_test_s, y_test)
    tree_rows[algo] = {"Accuracy": round(r["accuracy"], 4),
                        "Training Time (s)": round(r["train_time"], 5),
                        "Prediction Time (s)": round(r["predict_time"], 5)}

tree_table = pd.DataFrame(tree_rows)
print("\nKDTree vs BallTree:\n", tree_table)
save_table(tree_table.reset_index().rename(columns={"index": "Metric"}), "table_kdtree_vs_balltree.csv")

# 8. 5-fold cross validation for the best Naive Bayes and the best KNN
print("\nrunning 5-fold cross validation")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if best_nb_name == "Gaussian":
    nb_cv_scores = cross_val_score(GaussianNB(), X_train_s, y_train, cv=skf)
else:
    nb_model_cls = MultinomialNB if best_nb_name == "Multinomial" else BernoulliNB
    nb_cv_scores = cross_val_score(nb_model_cls(), X_train, y_train, cv=skf)

knn_cv_scores = cross_val_score(KNeighborsClassifier(**best_knn_params), X_train_s, y_train, cv=skf)

cv_table = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5, "Average"],
    f"Naive Bayes ({best_nb_name})": list(np.round(nb_cv_scores, 4)) + [round(nb_cv_scores.mean(), 4)],
    "Best KNN": list(np.round(knn_cv_scores, 4)) + [round(knn_cv_scores.mean(), 4)],
})
print("\ncross validation results:\n", cv_table)
save_table(cv_table, "table_cross_validation.csv")

fig, ax = plt.subplots(figsize=(6, 4))
folds = np.arange(1, 6)
ax.plot(folds, nb_cv_scores, marker="o", label=f"Naive Bayes ({best_nb_name})")
ax.plot(folds, knn_cv_scores, marker="s", label="Best KNN")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy")
ax.set_xticks(folds)
ax.legend()
save_plot(fig, "09_cross_validation_accuracy.png")

# 9. training / prediction time comparison
print("\ncomparing timing across models")
timing_rows = {
    "Gaussian NB": nb_results["Gaussian"],
    "Multinomial NB": nb_results["Multinomial"],
    "Bernoulli NB": nb_results["Bernoulli"],
    "Best KNN": best_knn_result,
}
timing_table = pd.DataFrame({
    name: {"Training(s)": round(r["train_time"], 5), "Prediction(s)": round(r["predict_time"], 5)}
    for name, r in timing_rows.items()
}).T
print("\nexperimental time analysis:\n", timing_table)
save_table(timing_table.reset_index().rename(columns={"index": "Algorithm"}), "table_experimental_time_analysis.csv")

# theoretical complexity, straight from the lab manual, just here for reference
theory_table = pd.DataFrame([
    {"Algorithm": "Naive Bayes", "Training": "O(n*d)", "Prediction": "O(d)"},
    {"Algorithm": "KNN (Brute)", "Training": "O(1)", "Prediction": "O(n*d)"},
    {"Algorithm": "KDTree", "Training": "O(n log n)", "Prediction": "O(log n) avg"},
    {"Algorithm": "BallTree", "Training": "O(n log n)", "Prediction": "O(log n) avg"},
])
save_table(theory_table, "table_theoretical_time_complexity.csv")

fig, ax = plt.subplots(figsize=(7, 4))
names = list(timing_rows.keys())
train_times = [timing_rows[n]["train_time"] for n in names]
ax.bar(names, train_times, color="cornflowerblue")
ax.set_ylabel("seconds"); ax.set_title("Training time comparison")
ax.tick_params(axis="x", rotation=20)
save_plot(fig, "10_training_time_comparison.png")

fig, ax = plt.subplots(figsize=(7, 4))
predict_times = [timing_rows[n]["predict_time"] for n in names]
ax.bar(names, predict_times, color="salmon")
ax.set_ylabel("seconds"); ax.set_title("Prediction time comparison")
ax.tick_params(axis="x", rotation=20)
save_plot(fig, "11_prediction_time_comparison.png")

fig, ax = plt.subplots(figsize=(8, 5))
all_names = list(nb_results.keys()) + [f"KNN k={k}" for k in k_values] + ["Best KNN (tuned)"]
all_acc = [nb_results[n]["accuracy"] for n in nb_results] + \
          [knn_results[k]["accuracy"] for k in k_values] + [best_knn_result["accuracy"]]
colors = ["indianred"] * 3 + ["cornflowerblue"] * len(k_values) + ["darkgreen"]
ax.bar(all_names, all_acc, color=colors)
ax.set_ylabel("Accuracy"); ax.set_title("Classifier comparison - test accuracy")
ax.tick_params(axis="x", rotation=60)
save_plot(fig, "12_classifier_comparison_bar_chart.png")

# best KNN gets its own confusion matrix too
fig, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_test, best_knn_result["pred"])
ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues")
ax.set_title(f"Best KNN - Confusion Matrix (acc={best_knn_result['accuracy']:.3f})")
save_plot(fig, "05b_confusion_matrix_best_knn.png")

# and one more ROC/PR pair with best KNN thrown in alongside the NB variants
fig, ax = plt.subplots(figsize=(6, 6))
for name, r in list(nb_results.items()):
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, label=f"{name} NB (AUC={r['roc_auc']:.3f})")
fpr, tpr, _ = roc_curve(y_test, best_knn_result["proba"])
ax.plot(fpr, tpr, label=f"Best KNN (AUC={best_knn_result['roc_auc']:.3f})", linewidth=2, color="black")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves - all models")
ax.legend()
save_plot(fig, "06b_roc_curve_all_models.png")

fig, ax = plt.subplots(figsize=(6, 6))
for name, r in list(nb_results.items()):
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    ax.plot(rec, prec, label=f"{name} NB")
prec, rec, _ = precision_recall_curve(y_test, best_knn_result["proba"])
ax.plot(rec, prec, label="Best KNN", linewidth=2, color="black")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves - all models")
ax.legend()
save_plot(fig, "07b_precision_recall_all_models.png")

# 10. a few extra things worth checking beyond the core comparison
print("\nrunning the additional tasks")

# how sensitive is accuracy to the train/test split ratio
split_rows = []
for test_size in [0.1, 0.2, 0.3]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    r = evaluate(KNeighborsClassifier(**best_knn_params), sc.transform(Xtr), ytr, sc.transform(Xte), yte)
    split_rows.append({"Test size": test_size, "Accuracy": round(r["accuracy"], 4)})
split_table = pd.DataFrame(split_rows)
print("\ndifferent train-test splits (best KNN):\n", split_table)
save_table(split_table, "table_train_test_splits.csv")

# euclidean vs manhattan distance
dist_rows = []
for metric in ["euclidean", "manhattan"]:
    r = evaluate(KNeighborsClassifier(n_neighbors=best_k, metric=metric), X_train_s, y_train, X_test_s, y_test)
    dist_rows.append({"Distance metric": metric, "Accuracy": round(r["accuracy"], 4)})
dist_table = pd.DataFrame(dist_rows)
print("\neuclidean vs manhattan distance (k=%d):\n" % best_k, dist_table)
save_table(dist_table, "table_distance_metric_comparison.csv")

# weighted KNN vs uniform KNN
weight_rows = []
for w in ["uniform", "distance"]:
    r = evaluate(KNeighborsClassifier(n_neighbors=best_k, weights=w), X_train_s, y_train, X_test_s, y_test)
    weight_rows.append({"Weights": w, "Accuracy": round(r["accuracy"], 4)})
weight_table = pd.DataFrame(weight_rows)
print("\nweighted vs uniform KNN (k=%d):\n" % best_k, weight_table)
save_table(weight_table, "table_weighted_knn_comparison.csv")

# 11. write out the analysis answers using the numbers computed above
answers = f"""Experiment 2 - Analysis Questions (auto-generated from results)

1. Which Naive Bayes variant performed best?
   {best_nb_name} Naive Bayes performed best, with accuracy={nb_results[best_nb_name]['accuracy']:.4f}
   and ROC-AUC={nb_results[best_nb_name]['roc_auc']:.4f}.

2. What is the optimal value of k?
   From the k-sweep, k={knn_table.loc[knn_table['Accuracy'].idxmax(), 'k']} gave the highest test accuracy
   ({knn_table['Accuracy'].max():.4f}). GridSearchCV independently selected
   k={grid.best_params_['n_neighbors']} as optimal using 5-fold CV.

3. Compare GridSearchCV and RandomizedSearchCV.
   GridSearchCV best CV accuracy={grid.best_score_:.4f} in {grid_time:.2f}s (exhaustive, {len(grid.cv_results_['params'])} combinations).
   RandomizedSearchCV best CV accuracy={rand_search.best_score_:.4f} in {rand_time:.2f}s (15 sampled combinations).
   RandomizedSearchCV is faster and found a {'similar' if abs(grid.best_score_-rand_search.best_score_)<0.01 else 'different'}
   quality solution, showing it is a good cheaper alternative when the search space is large.

4. Compare KDTree and BallTree.
   KDTree: accuracy={tree_table['kd_tree']['Accuracy']}, train={tree_table['kd_tree']['Training Time (s)']}s,
   predict={tree_table['kd_tree']['Prediction Time (s)']}s.
   BallTree: accuracy={tree_table['ball_tree']['Accuracy']}, train={tree_table['ball_tree']['Training Time (s)']}s,
   predict={tree_table['ball_tree']['Prediction Time (s)']}s.
   Both give identical/near-identical accuracy since they are exact search structures; they only
   differ in indexing speed, which matters more on higher-dimensional or larger datasets.

5. Compare theoretical and practical complexity.
   Theoretically Naive Bayes trains in O(n*d) and predicts in O(d), while KNN has ~O(1) training but
   O(n*d) brute-force prediction (or O(log n) average with a tree). In practice, Naive Bayes trained in
   {nb_results[best_nb_name]['train_time']:.5f}s and predicted in {nb_results[best_nb_name]['predict_time']:.5f}s,
   while Best KNN trained in {best_knn_result['train_time']:.5f}s and predicted in
   {best_knn_result['predict_time']:.5f}s - matching the theoretical expectation that KNN prediction
   is relatively much slower than Naive Bayes prediction.

6. Which classifier is preferred for large datasets?
   Naive Bayes is preferred for large datasets: its training and prediction cost scale linearly
   with features and are independent of the number of stored neighbors, unlike KNN, whose
   prediction cost grows with dataset size unless tree structures (KDTree/BallTree) are used,
   and even then it needs distance computations at inference time.
"""
with open(os.path.join(OUTDIR, "analysis_answers.txt"), "w") as f:
    f.write(answers)
print(answers)

print("\nExperiment 2 done, everything is in:", OUTDIR)
