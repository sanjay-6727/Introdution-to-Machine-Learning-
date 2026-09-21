import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------- Load & split ----------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign in sklearn's encoding

print("Dataset shape:", X.shape)
print("Class distribution:")
print(pd.Series(y).map({0: "Malignant", 1: "Benign"}).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(pipe, name):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name}: Test Acc={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return {"model": name, "pipe": pipe, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------- Bagging ----------------
print("\n" + "=" * 60)
print("BAGGING (base estimator: Decision Tree)")
print("=" * 60)

bagging_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    ))
])

bagging_grid = {
    "model__n_estimators": [10, 50, 100],
    "model__max_samples": [0.5, 0.7, 1.0],
    "model__max_features": [0.5, 0.7, 1.0]
}

bagging_search = GridSearchCV(bagging_pipe, bagging_grid, cv=cv, scoring="accuracy", n_jobs=-1)
bagging_search.fit(X_train, y_train)

print("Best params:", bagging_search.best_params_)
bagging_cv_results = pd.DataFrame(bagging_search.cv_results_)
bagging_table = bagging_cv_results.sort_values("mean_test_score", ascending=False).head(6)[
    ["param_model__n_estimators", "param_model__max_samples", "mean_test_score"]
]
print(bagging_table.to_string(index=False))

bagging_best = bagging_search.best_estimator_
bagging_f1_scores = cross_val_score(bagging_best, X_train, y_train, cv=cv, scoring="f1")
print("Best model CV F1 (5-fold):", bagging_f1_scores, "mean:", bagging_f1_scores.mean())

bagging_result = evaluate(bagging_best, "Bagging")

# ---------------- Boosting: AdaBoost ----------------
print("\n" + "=" * 60)
print("BOOSTING - ADABOOST")
print("=" * 60)

ada_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", AdaBoostClassifier(random_state=RANDOM_STATE))
])

ada_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.1, 1.0]
}

ada_search = GridSearchCV(ada_pipe, ada_grid, cv=cv, scoring="accuracy", n_jobs=-1)
ada_search.fit(X_train, y_train)
print("Best params:", ada_search.best_params_)

ada_cv_results = pd.DataFrame(ada_search.cv_results_)
ada_table = ada_cv_results.sort_values("mean_test_score", ascending=False).head(6)[
    ["param_model__n_estimators", "param_model__learning_rate", "mean_test_score"]
]
print(ada_table.to_string(index=False))

ada_best = ada_search.best_estimator_
ada_f1_scores = cross_val_score(ada_best, X_train, y_train, cv=cv, scoring="f1")
print("Best model CV F1 (5-fold):", ada_f1_scores, "mean:", ada_f1_scores.mean())

ada_result = evaluate(ada_best, "AdaBoost")

# ---------------- Boosting: Gradient Boosting ----------------
print("\n" + "=" * 60)
print("BOOSTING - GRADIENT BOOSTING")
print("=" * 60)

gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

gb_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__max_depth": [2, 3, 5]
}

gb_search = GridSearchCV(gb_pipe, gb_grid, cv=cv, scoring="accuracy", n_jobs=-1)
gb_search.fit(X_train, y_train)
print("Best params:", gb_search.best_params_)

gb_cv_results = pd.DataFrame(gb_search.cv_results_)
gb_table = gb_cv_results.sort_values("mean_test_score", ascending=False).head(6)[
    ["param_model__n_estimators", "param_model__learning_rate", "mean_test_score"]
]
print(gb_table.to_string(index=False))

gb_best = gb_search.best_estimator_
gb_f1_scores = cross_val_score(gb_best, X_train, y_train, cv=cv, scoring="f1")
print("Best model CV F1 (5-fold):", gb_f1_scores, "mean:", gb_f1_scores.mean())

gb_result = evaluate(gb_best, "Gradient Boosting")

# Pick the stronger boosting model to represent "Boosting" in the final comparison
boosting_result = ada_result if ada_result["f1"] >= gb_result["f1"] else gb_result
boosting_cv_f1 = ada_f1_scores if boosting_result["model"] == "AdaBoost" else gb_f1_scores

# ---------------- Stacking ----------------
print("\n" + "=" * 60)
print("STACKED ENSEMBLE (SVM + Naive Bayes + Decision Tree -> Logistic Regression)")
print("=" * 60)

base_learners = [
    ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ("nb", GaussianNB()),
    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))
]

stack_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5
    ))
])

stack_grid = {
    "model__svm__C": [0.1, 1, 10],
    "model__dt__max_depth": [3, 5, None]
}

stack_search = GridSearchCV(stack_pipe, stack_grid, cv=cv, scoring="accuracy", n_jobs=-1)
stack_search.fit(X_train, y_train)
print("Best params:", stack_search.best_params_)

stack_cv_results = pd.DataFrame(stack_search.cv_results_)
stack_table = stack_cv_results.sort_values("mean_test_score", ascending=False).head(6)[
    ["param_model__svm__C", "param_model__dt__max_depth", "mean_test_score"]
]
print(stack_table.to_string(index=False))

stack_best = stack_search.best_estimator_
stack_f1_scores = cross_val_score(stack_best, X_train, y_train, cv=cv, scoring="f1")
print("Best model CV F1 (5-fold):", stack_f1_scores, "mean:", stack_f1_scores.mean())

stack_result = evaluate(stack_best, "Stacked Ensemble")

# ---------------- Final comparison table ----------------
print("\n" + "=" * 100)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 100)

final_df = pd.DataFrame([
    {"Model": "Bagging", "Accuracy": bagging_result["accuracy"], "Precision": bagging_result["precision"],
     "Recall": bagging_result["recall"], "F1": bagging_result["f1"]},
    {"Model": "AdaBoost", "Accuracy": ada_result["accuracy"], "Precision": ada_result["precision"],
     "Recall": ada_result["recall"], "F1": ada_result["f1"]},
    {"Model": "Gradient Boosting", "Accuracy": gb_result["accuracy"], "Precision": gb_result["precision"],
     "Recall": gb_result["recall"], "F1": gb_result["f1"]},
    {"Model": "Stacked Ensemble", "Accuracy": stack_result["accuracy"], "Precision": stack_result["precision"],
     "Recall": stack_result["recall"], "F1": stack_result["f1"]},
])
print(final_df.round(4).to_string(index=False))

# ---------------- Confusion matrices ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, pipe) in zip(
    axes,
    [("Bagging", bagging_best), ("Boosting (" + boosting_result["model"] + ")", boosting_result["pipe"]),
     ("Stacked Ensemble", stack_best)]
):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Malignant", "Benign"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices_ensembles.png", dpi=150)
plt.close()

# ---------------- ROC curves ----------------
plt.figure(figsize=(6.5, 5.5))
for name, pipe in [("Bagging", bagging_best), (boosting_result["model"], boosting_result["pipe"]),
                    ("Stacked Ensemble", stack_best)]:
    y_score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Ensemble Models")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves_ensembles.png", dpi=150)
plt.close()

print("\nSaved: confusion_matrices_ensembles.png, roc_curves_ensembles.png")
