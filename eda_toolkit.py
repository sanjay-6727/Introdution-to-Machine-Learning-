"""
eda_toolkit.py - Simple common EDA function for any dataset.

ICS1512 Machine Learning Algorithms Laboratory

How to use:

    import pandas as pd
    from eda_toolkit import run_eda

    df = pd.read_csv("your_data.csv")
    run_eda(df, target="species", outdir="eda_outputs")

This will create grid images for:
    1. Histograms (distribution of each numeric column)
    2. Box plots (outliers)
    3. Violin plots (distribution shape)
    4. Standardization vs Normalization
    5. Outlier detection (Z-score and IQR)
    6. Missing values (heatmap + bar chart)
    7. Correlation heatmap
    8. Pair plot
    9. Q-Q plots (normality check)
   10. Count plots for categorical columns

Each plot is saved twice:
    - a .eps file at 600 dpi (for reports / printing)
    - a .png file at 150 dpi (for quickly viewing on screen)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# save every figure like this
DPI = 600
FORMAT = "eps"
PNG_DPI = 150


def save_fig(fig, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()

    eps_path = os.path.join(outdir, filename + "." + FORMAT)
    fig.savefig(eps_path, format=FORMAT, dpi=DPI, bbox_inches="tight")

    png_path = os.path.join(outdir, filename + ".png")
    fig.savefig(png_path, format="png", dpi=PNG_DPI, bbox_inches="tight")

    plt.close(fig)
    print("saved:", eps_path, "and", png_path)


def get_numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df, max_unique=15):
    cols = df.select_dtypes(exclude="number").columns.tolist()
    return [c for c in cols if df[c].nunique() <= max_unique]


# ---------------------------------------------------------
# 1. Histograms
# ---------------------------------------------------------
def plot_histograms(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histograms", fontsize=14)
    save_fig(fig, outdir, name + "_01_histograms")


# ---------------------------------------------------------
# 2. Box plots
# ---------------------------------------------------------
def plot_boxplots(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="lightgreen")
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Box Plots", fontsize=14)
    save_fig(fig, outdir, name + "_02_boxplots")


# ---------------------------------------------------------
# 3. Violin plots
# ---------------------------------------------------------
def plot_violinplots(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        sns.violinplot(y=df[col], ax=axes[i], color="plum")
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Violin Plots", fontsize=14)
    save_fig(fig, outdir, name + "_03_violinplots")


# ---------------------------------------------------------
# 4. Standardization vs Normalization
# ---------------------------------------------------------
def plot_scaling(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, col in enumerate(num_cols):
        x = df[col].dropna()

        # raw
        sns.histplot(x, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(col + " - raw")

        # standardized (z-score)
        z = (x - x.mean()) / x.std()
        sns.histplot(z, ax=axes[i, 1], color="orange")
        axes[i, 1].set_title(col + " - standardized")

        # normalized (min-max)
        m = (x - x.min()) / (x.max() - x.min())
        sns.histplot(m, ax=axes[i, 2], color="seagreen")
        axes[i, 2].set_title(col + " - normalized")

    fig.suptitle("Standardization vs Normalization", fontsize=14)
    save_fig(fig, outdir, name + "_04_scaling")


# ---------------------------------------------------------
# 5. Outlier detection (Z-score and IQR)
# ---------------------------------------------------------
def plot_outliers(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Z-score method: count outliers per column
    z_counts = []
    for col in num_cols:
        x = df[col].dropna()
        z = np.abs((x - x.mean()) / x.std())
        z_counts.append((z > 3).sum())

    axes[0].bar(num_cols, z_counts, color="salmon")
    axes[0].set_title("Outliers per feature (|Z-score| > 3)")
    axes[0].tick_params(axis="x", rotation=45)

    # IQR method: count outliers per column
    iqr_counts = []
    for col in num_cols:
        x = df[col].dropna()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_counts.append(((x < lower) | (x > upper)).sum())

    axes[1].bar(num_cols, iqr_counts, color="cornflowerblue")
    axes[1].set_title("Outliers per feature (IQR rule)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Outlier Detection", fontsize=14)
    save_fig(fig, outdir, name + "_05_outliers")


# ---------------------------------------------------------
# 6. Missing values
# ---------------------------------------------------------
def plot_missing_values(df, outdir, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # heatmap of missing values
    sns.heatmap(df.isnull(), cbar=False, ax=axes[0], cmap="Greys")
    axes[0].set_title("Missing Value Heatmap")

    # bar chart of missing percentage per column
    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(missing_pct) > 0:
        axes[1].bar(missing_pct.index, missing_pct.values, color="tomato")
        axes[1].set_ylabel("% missing")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[1].axis("off")
    axes[1].set_title("Missing Value Percentage")

    fig.suptitle("Missing Values", fontsize=14)
    save_fig(fig, outdir, name + "_06_missing_values")


# ---------------------------------------------------------
# 7. Correlation heatmap
# ---------------------------------------------------------
def plot_correlation(df, num_cols, outdir, name):
    if len(num_cols) < 2:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.6), max(6, len(num_cols) * 0.5)))
    corr = df[num_cols].corr()
    annotate = len(num_cols) <= 20
    sns.heatmap(corr, annot=annotate, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_fig(fig, outdir, name + "_07_correlation")


# ---------------------------------------------------------
# 8. Pair plot
# ---------------------------------------------------------
def plot_pairplot(df, num_cols, target, outdir, name):
    if len(num_cols) < 2:
        return
    cols = num_cols[:5]  # keep it simple, max 5 features
    if target is not None and target in df.columns:
        grid = sns.pairplot(df, vars=cols, hue=target)
    else:
        grid = sns.pairplot(df, vars=cols)
    grid.fig.suptitle("Pair Plot", y=1.02, fontsize=14)
    save_fig(grid.fig, outdir, name + "_08_pairplot")


# ---------------------------------------------------------
# 9. Q-Q plots (normality check)
# ---------------------------------------------------------
def plot_qqplots(df, num_cols, outdir, name):
    n = len(num_cols)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        x = df[col].dropna()
        stats.probplot(x, dist="norm", plot=axes[i])
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Q-Q Plots (Normality Check)", fontsize=14)
    save_fig(fig, outdir, name + "_09_qqplots")


# ---------------------------------------------------------
# 10. Categorical count plots
# ---------------------------------------------------------
def plot_categorical_counts(df, cat_cols, outdir, name):
    n = len(cat_cols)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cat_cols):
        sns.countplot(x=df[col], ax=axes[i], color="mediumpurple")
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Categorical Count Plots", fontsize=14)
    save_fig(fig, outdir, name + "_10_categorical")


# ---------------------------------------------------------
# Main function: run all EDA plots on any dataset
# ---------------------------------------------------------
def run_eda(df, target=None, outdir="eda_outputs", name="dataset", max_pairplot_cols=None):
    """
    Run standard EDA on any dataframe and save grid plots (EPS 600 dpi + PNG preview).

    df      : pandas DataFrame
    target  : name of the target/label column (optional, used for pair plot color)
    outdir  : folder to save the images in
    name    : prefix used for the saved file names
    """
    os.makedirs(outdir, exist_ok=True)

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    # target column should not be treated as a regular feature
    if target in num_cols:
        num_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)

    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    plot_histograms(df, num_cols, outdir, name)
    plot_boxplots(df, num_cols, outdir, name)
    plot_violinplots(df, num_cols, outdir, name)
    plot_scaling(df, num_cols, outdir, name)
    plot_outliers(df, num_cols, outdir, name)
    plot_missing_values(df, outdir, name)
    plot_correlation(df, num_cols, outdir, name)
    plot_pairplot(df, num_cols, target, outdir, name)
    plot_qqplots(df, num_cols, outdir, name)
    plot_categorical_counts(df, cat_cols, outdir, name)

    print("EDA complete. Files saved in:", outdir)


# ---------------------------------------------------------
# Demo run (uses the Iris dataset)
# ---------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)

    run_eda(df, target="species", outdir="eda_outputs", name="iris")
