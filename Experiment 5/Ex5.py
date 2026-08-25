# ============================================================
# ICS1512 - Machine Learning Algorithms Laboratory
# Experiment 5
# Decision Tree and Random Forest:
# A Comparative Classification Study
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ------------------------------------------------------------
# IMPORT YOUR EDA TOOLKIT
# ------------------------------------------------------------

from eda_toolkit import run_eda


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)
os.makedirs("outputs/decision_tree", exist_ok=True)
os.makedirs("outputs/random_forest", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


# ============================================================
# 2. LOAD WISCONSIN BREAST CANCER DATASET
# ============================================================

data = load_breast_cancer(as_frame=True)

df = data.frame.copy()

# Original sklearn target:
# 0 = malignant
# 1 = benign

df["diagnosis"] = df["target"].map({
    0: "M",
    1: "B"
})

# Remove numerical target column
df.drop(columns=["target"], inplace=True)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nClass distribution percentage:")
print(
    df["diagnosis"].value_counts(normalize=True) * 100
)

print("\nMissing values:")
print(df.isnull().sum().sum())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS USING PROVIDED TOOLKIT
# ============================================================

print("\n" + "=" * 70)
print("RUNNING EDA TOOLKIT")
print("=" * 70)

run_eda(
    df,
    target="diagnosis",
    outdir="outputs/eda",
    name="breast_cancer"
)

print("\nEDA completed.")
print("EDA files are available inside:")
print("outputs/eda/")


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["diagnosis"])

# Convert:
# M -> 1
# B -> 0

y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# ============================================================
# 5. TRAIN-TEST SPLIT - 80/20
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 6. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 7. DECISION TREE - HYPERPARAMETER SEARCH
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE HYPERPARAMETER TUNING")
print("=" * 70)

dt = DecisionTreeClassifier(
    random_state=42
)

dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt_grid = GridSearchCV(
    estimator=dt,
    param_grid=dt_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

dt_grid.fit(X_train, y_train)


# ============================================================
# 8. DECISION TREE CV RESULTS TABLE
# ============================================================

dt_results = pd.DataFrame(dt_grid.cv_results_)

dt_results_table = dt_results[
    [
        "param_criterion",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].copy()

dt_results_table.columns = [
    "Criterion",
    "Max Depth",
    "Min Samples Split",
    "Min Samples Leaf",
    "Avg CV Accuracy",
    "Std CV Accuracy",
    "Avg Train Accuracy"
]

dt_results_table["Avg CV Accuracy"] *= 100
dt_results_table["Std CV Accuracy"] *= 100
dt_results_table["Avg Train Accuracy"] *= 100

dt_results_table = dt_results_table.sort_values(
    by="Avg CV Accuracy",
    ascending=False
)

print("\nDecision Tree Cross-Fold Results:")
print(
    dt_results_table.head(20).to_string(index=False)
)

dt_results_table.to_csv(
    "outputs/results/decision_tree_cv_results.csv",
    index=False
)


# ============================================================
# 9. BEST DECISION TREE
# ============================================================

best_dt = dt_grid.best_estimator_

print("\nBest Decision Tree Parameters:")
print(dt_grid.best_params_)

print(
    "Best Decision Tree CV Accuracy: "
    f"{dt_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# 10. TRAIN BEST DECISION TREE
# ============================================================

best_dt.fit(X_train, y_train)

dt_pred = best_dt.predict(X_test)
dt_prob = best_dt.predict_proba(X_test)[:, 1]


# ============================================================
# 11. DECISION TREE TEST METRICS
# ============================================================

dt_accuracy = accuracy_score(y_test, dt_pred)

dt_precision = precision_score(
    y_test,
    dt_pred,
    zero_division=0
)

dt_recall = recall_score(
    y_test,
    dt_pred,
    zero_division=0
)

dt_f1 = f1_score(
    y_test,
    dt_pred,
    zero_division=0
)

dt_auc = roc_auc_score(
    y_test,
    dt_prob
)

print("\n" + "=" * 70)
print("DECISION TREE TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {dt_accuracy:.4f}")
print(f"Precision : {dt_precision:.4f}")
print(f"Recall    : {dt_recall:.4f}")
print(f"F1 Score  : {dt_f1:.4f}")
print(f"ROC-AUC   : {dt_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dt_pred,
        target_names=["Benign", "Malignant"]
    )
)


# ============================================================
# 12. DECISION TREE CONFUSION MATRIX
# ============================================================

dt_cm = confusion_matrix(
    y_test,
    dt_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/decision_tree/confusion_matrix.png",
    dpi=300
)

plt.savefig(
    "outputs/decision_tree/confusion_matrix.eps",
    dpi=600
)

plt.show()


# ============================================================
# 13. DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(24, 12))

plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=["Benign", "Malignant"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Best Decision Tree")

plt.savefig(
    "outputs/decision_tree/decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "outputs/decision_tree/decision_tree.eps",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. RANDOM FOREST - HYPERPARAMETER SEARCH
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 8, 10],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

rf_grid.fit(X_train, y_train)


# ============================================================
# 15. RANDOM FOREST CV RESULTS TABLE
# ============================================================

rf_results = pd.DataFrame(
    rf_grid.cv_results_
)

rf_results_table = rf_results[
    [
        "param_n_estimators",
        "param_max_depth",
        "param_max_features",
        "param_bootstrap",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].copy()

rf_results_table.columns = [
    "N Estimators",
    "Max Depth",
    "Max Features",
    "Bootstrap",
    "Avg CV Accuracy",
    "Std CV Accuracy",
    "Avg Train Accuracy"
]

rf_results_table["Avg CV Accuracy"] *= 100
rf_results_table["Std CV Accuracy"] *= 100
rf_results_table["Avg Train Accuracy"] *= 100

rf_results_table = rf_results_table.sort_values(
    by="Avg CV Accuracy",
    ascending=False
)

print("\nRandom Forest Cross-Fold Results:")
print(
    rf_results_table.head(20).to_string(index=False)
)

rf_results_table.to_csv(
    "outputs/results/random_forest_cv_results.csv",
    index=False
)


# ============================================================
# 16. BEST RANDOM FOREST
# ============================================================

best_rf = rf_grid.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

print(
    "Best Random Forest CV Accuracy: "
    f"{rf_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# 17. TRAIN BEST RANDOM FOREST
# ============================================================

best_rf.fit(X_train, y_train)

rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]


# ============================================================
# 18. RANDOM FOREST TEST METRICS
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

print("\n" + "=" * 70)
print("RANDOM FOREST TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Benign", "Malignant"]
    )
)


# ============================================================
# 19. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/random_forest/confusion_matrix.png",
    dpi=300
)

plt.savefig(
    "outputs/random_forest/confusion_matrix.eps",
    dpi=600
)

plt.show()


# ============================================================
# 20. ROC CURVE COMPARISON
# ============================================================

dt_fpr, dt_tpr, dt_thresholds = roc_curve(
    y_test,
    dt_prob
)

rf_fpr, rf_tpr, rf_thresholds = roc_curve(
    y_test,
    rf_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    dt_fpr,
    dt_tpr,
    label=f"Decision Tree (AUC = {dt_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Decision Tree vs Random Forest")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/results/roc_comparison.png",
    dpi=300
)

plt.savefig(
    "outputs/results/roc_comparison.eps",
    dpi=600
)

plt.show()


# ============================================================
# 21. FOLD-WISE CROSS VALIDATION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION PERFORMANCE COMPARISON")
print("=" * 70)

dt_fold_scores = cross_val_score(
    best_dt,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

rf_fold_scores = cross_val_score(
    best_rf,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

fold_table = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "Fold 1": [
        dt_fold_scores[0],
        rf_fold_scores[0]
    ],

    "Fold 2": [
        dt_fold_scores[1],
        rf_fold_scores[1]
    ],

    "Fold 3": [
        dt_fold_scores[2],
        rf_fold_scores[2]
    ],

    "Fold 4": [
        dt_fold_scores[3],
        rf_fold_scores[3]
    ],

    "Fold 5": [
        dt_fold_scores[4],
        rf_fold_scores[4]
    ],

    "Average": [
        dt_fold_scores.mean(),
        rf_fold_scores.mean()
    ]
})

print(
    fold_table.to_string(index=False)
)

fold_table.to_csv(
    "outputs/results/fold_comparison.csv",
    index=False
)


# ============================================================
# 22. FINAL MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ],

    "Precision": [
        dt_precision,
        rf_precision
    ],

    "Recall": [
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        dt_f1,
        rf_f1
    ],

    "ROC-AUC": [
        dt_auc,
        rf_auc
    ]
})

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(index=False)
)

comparison.to_csv(
    "outputs/results/model_comparison.csv",
    index=False
)


# ============================================================
# 23. METRICS BAR CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    comparison.iloc[0][metrics],
    width,
    label="Decision Tree"
)

plt.bar(
    x + width / 2,
    comparison.iloc[1][metrics],
    width,
    label="Random Forest"
)

plt.xticks(
    x,
    metrics,
    rotation=20
)

plt.ylabel("Score")

plt.title(
    "Decision Tree vs Random Forest - Performance Comparison"
)

plt.ylim(0, 1.1)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/results/model_comparison.png",
    dpi=300
)

plt.savefig(
    "outputs/results/model_comparison.eps",
    dpi=600
)

plt.show()


# ============================================================
# 24. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 10 RANDOM FOREST FEATURE IMPORTANCES")
print("=" * 70)

print(
    feature_importance.head(10).to_string(index=False)
)

feature_importance.to_csv(
    "outputs/results/random_forest_feature_importance.csv",
    index=False
)


plt.figure(figsize=(10, 6))

top_features = feature_importance.head(10)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.title(
    "Top 10 Random Forest Feature Importances"
)

plt.tight_layout()

plt.savefig(
    "outputs/results/feature_importance.png",
    dpi=300
)

plt.savefig(
    "outputs/results/feature_importance.eps",
    dpi=600
)

plt.show()


# ============================================================
# 25. SAVE BEST PARAMETERS
# ============================================================

with open(
    "outputs/results/best_parameters.txt",
    "w"
) as f:

    f.write("DECISION TREE\n")
    f.write("=" * 50 + "\n")
    f.write(str(dt_grid.best_params_) + "\n")
    f.write(
        f"Best CV Accuracy: "
        f"{dt_grid.best_score_:.4f}\n\n"
    )

    f.write("RANDOM FOREST\n")
    f.write("=" * 50 + "\n")
    f.write(str(rf_grid.best_params_) + "\n")
    f.write(
        f"Best CV Accuracy: "
        f"{rf_grid.best_score_:.4f}\n"
    )


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 5 COMPLETED")
print("=" * 70)

print("\nDecision Tree:")
print(f"  Best Parameters : {dt_grid.best_params_}")
print(f"  CV Accuracy     : {dt_grid.best_score_:.4f}")
print(f"  Test Accuracy   : {dt_accuracy:.4f}")
print(f"  Test F1         : {dt_f1:.4f}")
print(f"  Test ROC-AUC    : {dt_auc:.4f}")

print("\nRandom Forest:")
print(f"  Best Parameters : {rf_grid.best_params_}")
print(f"  CV Accuracy     : {rf_grid.best_score_:.4f}")
print(f"  Test Accuracy   : {rf_accuracy:.4f}")
print(f"  Test F1         : {rf_f1:.4f}")
print(f"  Test ROC-AUC    : {rf_auc:.4f}")

print("\nAll outputs saved inside:")
print("outputs/")
