"""
Experiment 1: Working with Python packages - Numpy, Scipy, Scikit-Learn, Matplotlib
ICS1512 Machine Learning Algorithms Laboratory

Runs the standard ML workflow (load -> EDA -> preprocess -> feature selection ->
split -> train -> evaluate) on 5 datasets: Iris, Loan Amount Prediction,
Predicting Diabetes, Email Spam (Spambase) and Handwritten Digits (MNIST style).
Also does a quick tour of numpy/pandas/scipy/matplotlib at the top since that
was asked for separately.

Images go into the "outputs" folder next to this script.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2, VarianceThreshold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.datasets import load_iris, load_digits

# lets us import eda_toolkit.py from the parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eda_toolkit import run_eda

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "outputs")
DATADIR = os.path.join(HERE, "data")
os.makedirs(OUTDIR, exist_ok=True)

results_table = []   # one row per dataset, used for the summary table at the end


def save_plot(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


def pick_top_features_for_eda(X, y, max_features=12, classification=True):
    """Picks the most informative columns so the EDA grids don't get too cluttered."""
    if X.shape[1] <= max_features:
        return X.columns.tolist()
    score_func = f_classif if classification else f_regression
    selector = SelectKBest(score_func=score_func, k=max_features)
    selector.fit(X.fillna(X.median(numeric_only=True)), y)
    scores = pd.Series(selector.scores_, index=X.columns)
    return scores.sort_values(ascending=False).head(max_features).index.tolist()


# Part 0 - quick tour of the core libraries before touching any dataset
print("\nPart 0 - library tour")

# numpy: basic array stuff
arr = np.arange(1, 13).reshape(3, 4)
print("array:\n", arr)
print("sum/mean/std:", arr.sum(), arr.mean(), arr.std())
print("transpose:\n", arr.T)
print("matrix multiply:\n", arr @ arr.T)

# pandas: dataframe + groupby
demo_df = pd.DataFrame({
    "product": ["A", "B", "A", "B", "C"],
    "sales": [100, 150, 200, 130, 90],
})
print("\ndataframe:\n", demo_df)
print("groupby mean:\n", demo_df.groupby("product")["sales"].mean())

# scipy: describe + a quick t-test
sample = np.random.default_rng(0).normal(loc=50, scale=5, size=200)
print("\nscipy describe:", stats.describe(sample))
t_stat, p_value = stats.ttest_1samp(sample, popmean=50)
print("one-sample t-test vs mean=50 -> t=%.3f p=%.3f" % (t_stat, p_value))

# scikit-learn: tiny pipeline just to show the fit/predict API
iris_demo = load_iris()
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    iris_demo.data, iris_demo.target, test_size=0.3, random_state=0)
demo_model = KNeighborsClassifier(n_neighbors=5).fit(Xd_train, yd_train)
print("\nKNN demo accuracy on Iris:",
      accuracy_score(yd_test, demo_model.predict(Xd_test)))

# matplotlib: line, scatter, bar, histogram in one grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), color="steelblue")
axes[0, 0].set_title("Matplotlib line plot")
axes[0, 1].scatter(iris_demo.data[:, 0], iris_demo.data[:, 1],
                    c=iris_demo.target, cmap="viridis")
axes[0, 1].set_title("Matplotlib scatter plot")
axes[1, 0].bar(demo_df["product"], demo_df["sales"], color="coral")
axes[1, 0].set_title("Matplotlib bar plot")
axes[1, 1].hist(sample, bins=20, color="seagreen")
axes[1, 1].set_title("Matplotlib histogram")
fig.suptitle("Library Exploration Demo", fontsize=14)
save_plot(fig, "00_library_exploration_demo.png")


# Part 1 - Iris (supervised, multiclass classification)
print("\nPart 1 - Iris")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df = df.drop(columns=["target"])

print("shape:", df.shape)
print("missing values:\n", df.isna().sum())

run_eda(df, target="species", outdir=os.path.join(OUTDIR, "iris_eda"), name="iris")

X = df.drop(columns=["species"])
y = df["species"]

# ANOVA F-test to see which petal/sepal measurements separate the species best
selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(X, y)
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(scores.index, scores.values, color="teal")
ax.set_title("Iris - ANOVA F-test feature scores (SelectKBest)")
ax.set_xlabel("F-score")
save_plot(fig, "iris_feature_selection.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print("Iris Logistic Regression test accuracy:", round(acc, 4))

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=model.classes_).plot(ax=ax, cmap="Blues")
ax.set_title(f"Iris - Confusion Matrix (accuracy={acc:.2f})")
save_plot(fig, "iris_confusion_matrix.png")

results_table.append({
    "Dataset": "Iris Dataset",
    "Type of ML task": "Supervised - Multiclass Classification",
    "Feature Selection Technique": "ANOVA F-test (SelectKBest)",
    "Suitable ML Algorithm": "Logistic Regression / KNN / Decision Tree",
    "Baseline model used here": "Logistic Regression",
    "Test Accuracy": round(acc, 4),
})


# Part 2 - Loan Amount Prediction (supervised, regression)
print("\nPart 2 - Loan Amount Prediction")

loan = pd.read_csv(os.path.join(DATADIR, "loan_train.csv"))
loan = loan.drop(columns=["Customer ID", "Name", "Property ID"])
target_col = "Loan Sanction Amount (USD)"
loan = loan.dropna(subset=[target_col])   # can't train on a row with no target

print("shape:", loan.shape)
print("missing values (top 5):\n", loan.isna().sum().sort_values(ascending=False).head())

# median for numeric gaps, mode for categorical gaps
num_cols = loan.select_dtypes(include="number").columns.tolist()
num_cols.remove(target_col) if target_col in num_cols else None
cat_cols = loan.select_dtypes(exclude="number").columns.tolist()

for c in num_cols:
    loan[c] = loan[c].fillna(loan[c].median())
for c in cat_cols:
    loan[c] = loan[c].fillna(loan[c].mode()[0])

# label encode the categoricals so a plain LinearRegression can use them
loan_encoded = loan.copy()
for c in cat_cols:
    loan_encoded[c] = LabelEncoder().fit_transform(loan_encoded[c])

eda_cols = pick_top_features_for_eda(loan_encoded.drop(columns=[target_col]),
                                     loan_encoded[target_col], max_features=10,
                                     classification=False)
run_eda(loan_encoded[eda_cols + [target_col]], target=None,
        outdir=os.path.join(OUTDIR, "loan_eda"), name="loan")

X = loan_encoded.drop(columns=[target_col])
y = loan_encoded[target_col]

# F-regression is basically ANOVA's regression cousin
selector = SelectKBest(score_func=f_regression, k=8)
selector.fit(X, y)
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(scores.index, scores.values, color="darkorange")
ax.set_title("Loan Amount - F-regression feature scores (SelectKBest)")
ax.set_xlabel("F-score")
save_plot(fig, "loan_feature_selection.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = LinearRegression().fit(X_train_s, y_train)
pred = model.predict(X_test_s)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"Loan Amount Linear Regression -> MAE={mae:.2f}  R2={r2:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, pred, s=10, alpha=0.4, color="steelblue")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual loan amount")
ax.set_ylabel("Predicted loan amount")
ax.set_title(f"Loan Amount - Predicted vs Actual (R2={r2:.2f})")
save_plot(fig, "loan_predicted_vs_actual.png")

results_table.append({
    "Dataset": "Loan Amount Prediction",
    "Type of ML task": "Supervised - Regression",
    "Feature Selection Technique": "F-regression (SelectKBest)",
    "Suitable ML Algorithm": "Linear/Ridge Regression / Random Forest Regressor",
    "Baseline model used here": "Linear Regression",
    "Test Accuracy": f"R2={r2:.4f}",
})


# Part 3 - Predicting Diabetes (supervised, binary classification)
print("\nPart 3 - Predicting Diabetes")

diab = pd.read_csv(os.path.join(DATADIR, "diabetes.csv"))
print("shape:", diab.shape)
print("missing values:", diab.isna().sum().sum(), "(none technically, but 0s in some medical columns are really missing data)")

run_eda(diab, target="Outcome", outdir=os.path.join(OUTDIR, "diabetes_eda"), name="diabetes")

X = diab.drop(columns=["Outcome"])
y = diab["Outcome"]

# chi-square needs non-negative features, which these medical measurements are
selector = SelectKBest(score_func=chi2, k=5)
selector.fit(X, y)
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(scores.index, scores.values, color="indianred")
ax.set_title("Diabetes - Chi-square feature scores (SelectKBest)")
ax.set_xlabel("Chi2 score")
save_plot(fig, "diabetes_feature_selection.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print("Diabetes Random Forest test accuracy:", round(acc, 4))

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues")
ax.set_title(f"Diabetes - Confusion Matrix (accuracy={acc:.2f})")
save_plot(fig, "diabetes_confusion_matrix.png")

results_table.append({
    "Dataset": "Predicting Diabetes",
    "Type of ML task": "Supervised - Binary Classification",
    "Feature Selection Technique": "Chi-square test (SelectKBest)",
    "Suitable ML Algorithm": "Logistic Regression / Random Forest Classifier",
    "Baseline model used here": "Random Forest Classifier",
    "Test Accuracy": round(acc, 4),
})


# Part 4 - Email Spam Classification / Spambase (supervised, binary classification)
print("\nPart 4 - Email Spam Classification (Spambase)")

spam = pd.read_csv(os.path.join(DATADIR, "spambase_csv.csv"))
print("shape:", spam.shape)
print("missing values:", spam.isna().sum().sum())

X = spam.drop(columns=["class"])
y = spam["class"]

eda_cols = pick_top_features_for_eda(X, y, max_features=10, classification=True)
run_eda(spam[eda_cols + ["class"]], target="class",
        outdir=os.path.join(OUTDIR, "spambase_eda"), name="spambase")

# word frequencies are non-negative, so chi-square applies here too
selector = SelectKBest(score_func=chi2, k=10)
selector.fit(X, y)
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(scores.index, scores.values, color="purple")
ax.set_title("Spambase - Top Chi-square feature scores (SelectKBest)")
ax.set_xlabel("Chi2 score")
save_plot(fig, "spambase_feature_selection.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = LogisticRegression(max_iter=2000).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print("Spambase Logistic Regression test accuracy:", round(acc, 4))

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues")
ax.set_title(f"Spambase - Confusion Matrix (accuracy={acc:.2f})")
save_plot(fig, "spambase_confusion_matrix.png")

results_table.append({
    "Dataset": "Classification of Email Spam",
    "Type of ML task": "Supervised - Binary Classification",
    "Feature Selection Technique": "Chi-square test (SelectKBest)",
    "Suitable ML Algorithm": "Naive Bayes / SVM / Logistic Regression",
    "Baseline model used here": "Logistic Regression",
    "Test Accuracy": round(acc, 4),
})


# Part 5 - Handwritten Digits / MNIST style (supervised, multiclass classification)
print("\nPart 5 - Handwritten Character Recognition (Digits)")

digits = load_digits(as_frame=True)
df_digits = digits.frame.copy()
df_digits["digit"] = digits.target
df_digits = df_digits.drop(columns=["target"])
print("shape:", df_digits.shape, "(8x8 pixel images flattened to 64 features)")
print("missing values:", df_digits.isna().sum().sum())

X = df_digits.drop(columns=["digit"])
y = df_digits["digit"]

# a lot of the corner pixels are always black (0) since digits sit centered in the frame,
# variance threshold is a natural way to drop those dead pixels
vt = VarianceThreshold(threshold=1.0)
vt.fit(X)
kept = X.columns[vt.get_support()]
fig, ax = plt.subplots(figsize=(6, 4))
var_series = X.var().sort_values(ascending=False)
ax.bar(range(len(var_series)), var_series.values, color="slateblue")
ax.axhline(1.0, color="red", linestyle="--", label="variance threshold = 1.0")
ax.set_title(f"Digits - Pixel variance (kept {len(kept)}/{X.shape[1]} pixels)")
ax.set_xlabel("pixel feature (sorted by variance)")
ax.set_ylabel("variance")
ax.legend()
save_plot(fig, "digits_feature_selection.png")

eda_cols = pick_top_features_for_eda(X, y, max_features=10, classification=True)
run_eda(df_digits[eda_cols + ["digit"]], target=None,
        outdir=os.path.join(OUTDIR, "digits_eda"), name="digits")

# quick look at what the actual images look like
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"label={digits.target[i]}")
    ax.axis("off")
fig.suptitle("Sample handwritten digit images")
save_plot(fig, "digits_sample_images.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print("Digits KNN test accuracy:", round(acc, 4))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Digits - Confusion Matrix (accuracy={acc:.2f})")
save_plot(fig, "digits_confusion_matrix.png")

results_table.append({
    "Dataset": "Handwritten Character Recognition / MNIST",
    "Type of ML task": "Supervised - Multiclass Classification",
    "Feature Selection Technique": "Variance Threshold",
    "Suitable ML Algorithm": "KNN / SVM / CNN (deep learning)",
    "Baseline model used here": "K-Nearest Neighbors",
    "Test Accuracy": round(acc, 4),
})


# Part 6 - put it all together in one summary table
print("\nPart 6 - summary table")

summary = pd.DataFrame(results_table)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUTDIR, "ml_task_summary_table.csv"), index=False)
print("\nsaved summary table to:", os.path.join(OUTDIR, "ml_task_summary_table.csv"))

print("\nExperiment 1 done, everything is in:", OUTDIR)
