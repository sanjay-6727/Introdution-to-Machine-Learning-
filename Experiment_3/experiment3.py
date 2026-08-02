"""
Experiment 3: Regression Analysis using Linear and Regularized Models
ICS1512 Machine Learning Algorithms Laboratory

Dataset: Predict Loan Amount Data (Kaggle - phileinsophos/predict-loan-amount-data)
Target : Loan Sanction Amount (USD)

Steps:
    1. Load and preprocess the dataset
    2. Exploratory Data Analysis
    3. Train Linear, Ridge, Lasso, Elastic Net regression models
    4. Tune Ridge/Lasso/Elastic Net with GridSearchCV (5-fold CV)
    5. Evaluate with MAE, MSE, RMSE, R2 (cross-validation and test set)
    6. Analyze overfitting/underfitting and bias-variance trade-off
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eda_toolkit import run_eda

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "outputs")
DATADIR = os.path.join(HERE, "data")
os.makedirs(OUTDIR, exist_ok=True)

TARGET = "Loan Sanction Amount (USD)"


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


# 1. load and clean the data
print("loading data")
df = pd.read_csv(os.path.join(DATADIR, "train.csv"))
print("raw shape:", df.shape)

# these columns are just identifiers, no predictive value
df = df.drop(columns=["Customer ID", "Name", "Property ID"])

# some rows have no target, and some have -999 which is clearly a placeholder
# rather than a real sanctioned amount, both get dropped
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] != -999]
print("shape after removing missing/sentinel targets:", df.shape)

num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols.remove(TARGET)
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
print("numeric features:", num_cols)
print("categorical features:", cat_cols)
print("missing values before imputation:\n", df.isna().sum()[df.isna().sum() > 0])

# median for numeric gaps, mode for categorical gaps
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# one-hot encode the categoricals for the regression models
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("shape after one-hot encoding:", df_encoded.shape)

# 2. exploratory data analysis
print("\nrunning EDA")
run_eda(df[num_cols + [TARGET]], target=None,
        outdir=os.path.join(OUTDIR, "eda"), name="loan")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df[TARGET], bins=50, color="steelblue", edgecolor="white")
ax.set_xlabel("Loan Sanction Amount (USD)")
ax.set_ylabel("count")
ax.set_title("Target variable distribution")
save_plot(fig, "01_target_distribution.png")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
scatter_feats = ["Income (USD)", "Loan Amount Request (USD)", "Credit Score", "Property Price"]
for ax, feat in zip(axes.ravel(), scatter_feats):
    ax.scatter(df[feat], df[TARGET], s=5, alpha=0.3, color="teal")
    ax.set_xlabel(feat); ax.set_ylabel(TARGET)
fig.suptitle("Feature vs Target scatter plots")
save_plot(fig, "02_feature_vs_target_scatter.png")

# 3. standardize and split
print("\npreprocessing for modeling")
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train size: {len(X_train)}, test size: {len(X_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}


# 4. baseline Linear Regression
print("\ntraining baseline Linear Regression")
t0 = time.perf_counter()
linreg = LinearRegression().fit(X_train_s, y_train)
linreg_train_time = time.perf_counter() - t0
linreg_pred = linreg.predict(X_test_s)
print("Linear Regression test metrics:", regression_metrics(y_test, linreg_pred))

# 5. Ridge, Lasso, Elastic Net, each tuned with GridSearchCV
print("\ntuning Ridge, Lasso and Elastic Net")

param_grids = {
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=50000), {"alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "Elastic Net": (ElasticNet(max_iter=50000),
                     {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8]}),
}

tuned_models = {"Linear Regression": linreg}
tuning_rows = []
train_times = {"Linear Regression": linreg_train_time}

for name, (estimator, grid) in param_grids.items():
    t0 = time.perf_counter()
    search = GridSearchCV(estimator, grid, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train_s, y_train)
    elapsed = time.perf_counter() - t0
    tuned_models[name] = search.best_estimator_
    train_times[name] = elapsed
    tuning_rows.append({"Model": name, "Search Method": "Grid Search",
                         "Best Parameters": str(search.best_params_),
                         "Best CV R2": round(search.best_score_, 4)})
    print(f"{name}: best params={search.best_params_}  best CV R2={search.best_score_:.4f}  ({elapsed:.2f}s)")

tuning_table = pd.DataFrame(tuning_rows)
save_table(tuning_table, "table1_hyperparameter_tuning_summary.csv")

# 6. cross-validation performance for all four models
print("\nrunning 5-fold cross validation")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "MAE": make_scorer(mean_absolute_error),
    "MSE": make_scorer(mean_squared_error),
    "R2": make_scorer(r2_score),
}

cv_rows = []
for name, model in tuned_models.items():
    cvres = cross_validate(model, X_train_s, y_train, cv=kf, scoring=scoring)
    mae = cvres["test_MAE"].mean()
    mse = cvres["test_MSE"].mean()
    cv_rows.append({"Model": name, "MAE": round(mae, 2), "MSE": round(mse, 2),
                     "RMSE": round(np.sqrt(mse), 2), "R2": round(cvres["test_R2"].mean(), 4)})

cv_table = pd.DataFrame(cv_rows)
print(cv_table)
save_table(cv_table, "table2_cross_validation_performance.csv")

# 7. test set performance for all four models
print("\nchecking test set performance")
test_rows = []
test_preds = {}
for name, model in tuned_models.items():
    pred = model.predict(X_test_s)
    test_preds[name] = pred
    m = regression_metrics(y_test, pred)
    test_rows.append({"Model": name, "MAE": round(m["MAE"], 2), "MSE": round(m["MSE"], 2),
                       "RMSE": round(m["RMSE"], 2), "R2": round(m["R2"], 4),
                       "Training Time (s)": round(train_times[name], 4)})

test_table = pd.DataFrame(test_rows)
print(test_table)
save_table(test_table, "table3_test_set_performance.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(test_table["Model"], test_table["RMSE"], color="salmon")
axes[0].set_ylabel("RMSE"); axes[0].set_title("Test RMSE by model")
axes[0].tick_params(axis="x", rotation=20)
axes[1].bar(test_table["Model"], test_table["R2"], color="seagreen")
axes[1].set_ylabel("R2"); axes[1].set_title("Test R2 by model")
axes[1].tick_params(axis="x", rotation=20)
fig.suptitle("Test Set Performance Comparison")
save_plot(fig, "07_test_performance_comparison.png")

# 8. predicted vs actual, and the residuals
print("\nlooking at predicted vs actual and residuals")
best_name = test_table.loc[test_table["R2"].idxmax(), "Model"]
best_pred = test_preds[best_name]
print("best model on test set:", best_name)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, best_pred, s=8, alpha=0.35, color="steelblue")
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
ax.set_title(f"Predicted vs Actual - {best_name} (R2={test_table['R2'].max():.3f})")
save_plot(fig, "03_predicted_vs_actual.png")

residuals = y_test - best_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(best_pred, residuals, s=8, alpha=0.35, color="darkorange")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
axes[0].set_title(f"Residual plot - {best_name}")
axes[1].hist(residuals, bins=50, color="darkorange", edgecolor="white")
axes[1].set_xlabel("Residual"); axes[1].set_title("Residual distribution")
save_plot(fig, "04_residual_plot.png")

# 9. training vs validation error across Ridge's alpha, to see over/underfitting
print("\nsweeping Ridge alpha to look at training vs validation error")
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_errs, val_errs = [], []
for a in alphas:
    model = Ridge(alpha=a)
    cvres = cross_validate(model, X_train_s, y_train, cv=kf,
                            scoring=make_scorer(mean_squared_error),
                            return_train_score=True)
    train_errs.append(cvres["train_score"].mean())
    val_errs.append(cvres["test_score"].mean())

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(alphas, train_errs, marker="o", label="Training MSE")
ax.plot(alphas, val_errs, marker="s", label="Validation MSE")
ax.set_xscale("log")
ax.set_xlabel("Ridge alpha (regularization strength)")
ax.set_ylabel("Mean Squared Error")
ax.set_title("Training vs Validation error across regularization strength")
ax.legend()
save_plot(fig, "05_training_vs_validation_error.png")

# 10. how do coefficients change across the four models
print("\ncomparing coefficients across models")
coef_df = pd.DataFrame({name: model.coef_ for name, model in tuned_models.items()},
                        index=feature_names)
# just look at the 3 biggest coefficients from the plain Linear model
top3 = coef_df["Linear Regression"].abs().sort_values(ascending=False).head(3).index.tolist()
coef_table = coef_df.loc[top3].reset_index().rename(columns={"index": "Feature"})
coef_table["Feature"] = [f"Feature {i+1}: {name}" for i, name in enumerate(top3)]
print(coef_table.round(2))
save_table(coef_table.round(2), "table4_coefficient_comparison.csv")

# keep the full coefficient table around too, just in case
coef_df.round(3).reset_index().rename(columns={"index": "Feature"}).to_csv(
    os.path.join(OUTDIR, "coefficients_full.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(top3))
width = 0.2
for i, name in enumerate(tuned_models.keys()):
    ax.bar(x + i * width, coef_df.loc[top3, name].values, width=width, label=name)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([t[:25] for t in top3], rotation=20, ha="right")
ax.set_ylabel("Coefficient value")
ax.set_title("Effect of regularization on coefficients (top 3 features)")
ax.legend()
save_plot(fig, "06_coefficient_comparison.png")

# how many coefficients did Lasso actually zero out
lasso_zero = int((coef_df["Lasso"].abs() < 1e-6).sum())
print(f"Lasso set {lasso_zero} / {len(feature_names)} coefficients to (near) zero.")

# 11. write up the overfitting/underfitting and bias-variance analysis
best_ridge_alpha = tuning_rows[0]["Best Parameters"]
analysis = f"""Experiment 3 - Overfitting/Underfitting and Bias-Variance Analysis
(auto-generated from the results computed above)

Overfitting and Underfitting Analysis
--------------------------------------
- Training vs validation MSE (see 05_training_vs_validation_error.png): at very small
  Ridge alpha the model fits the training data closely (low training error) but validation
  error is higher - a sign of overfitting. As alpha increases, both errors converge and then
  rise together once the model becomes too constrained (underfitting) at very large alpha.
- Best Ridge alpha found by GridSearchCV: {tuning_rows[0]['Best Parameters']}
- Best Lasso alpha found by GridSearchCV: {tuning_rows[1]['Best Parameters']}
- Best Elastic Net params found by GridSearchCV: {tuning_rows[2]['Best Parameters']}
- Generalization after tuning: cross-validation R2 for the tuned models
  ({', '.join(f"{r['Model']}={r['Best CV R2']}" for r in tuning_rows)}) compared with plain
  Linear Regression's test R2={test_table.loc[test_table['Model']=='Linear Regression','R2'].values[0]}
  shows regularization {"improved" if max(r["Best CV R2"] for r in tuning_rows) > test_table.loc[test_table['Model']=='Linear Regression','R2'].values[0] else "did not noticeably improve"}
  generalization on this dataset.

Bias-Variance Analysis
------------------------
- Linear Regression has low bias (it fits all {len(feature_names)} features without penalty) but can
  have higher variance on noisy/collinear features, since coefficients are unconstrained.
- Ridge Regression reduces variance by shrinking coefficient magnitudes (see coefficient comparison
  plot) without setting them to zero, trading a small increase in bias for a reduction in variance.
- Lasso Regression performs feature selection: it drove {lasso_zero} out of {len(feature_names)}
  coefficients to exactly zero, producing a sparser and more interpretable model, at the cost
  of some bias if a truly useful feature gets zeroed out.
- Elastic Net (l1_ratio mixes L1 and L2) sits between Ridge and Lasso, offering some sparsity
  while still handling correlated features more gracefully than pure Lasso.

Conclusion
----------
Best performing model on the held-out test set: {best_name}
(Test RMSE={test_table.loc[test_table['Model']==best_name,'RMSE'].values[0]},
 Test R2={test_table.loc[test_table['Model']==best_name,'R2'].values[0]})
This suggests that for the Loan Amount dataset, {"regularization gives a measurable benefit over plain Linear Regression" if best_name != "Linear Regression" else "the plain Linear Regression baseline was already close to optimal, and regularization mainly helped with coefficient stability rather than raw accuracy"}.
"""
with open(os.path.join(OUTDIR, "overfitting_bias_variance_analysis.txt"), "w") as f:
    f.write(analysis)
print(analysis)

print("\nExperiment 3 done, everything is in:", OUTDIR)
