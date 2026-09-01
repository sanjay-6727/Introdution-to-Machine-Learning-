import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from xgboost import XGBClassifier


df = pd.read_csv("pca_classification_clean_1000.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["target_class"].value_counts())




X = df.drop(columns=["target", "target_class"])
y = df["target_class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


models = {

    "SVM": (
        SVC(probability=True),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
    ),

    "Naive Bayes": (
        GaussianNB(),
        {
            "model__var_smoothing": [
                1e-11,
                1e-10,
                1e-9,
                1e-8,
                1e-7
            ]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
    ),

    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 5]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        ),
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    )
}


def run_experiment(model, params, use_pca):

    steps = [
        ("scaler", StandardScaler())
    ]

    if use_pca:
        steps.append(
            ("pca", PCA())
        )

    steps.append(
        ("model", model)
    )

    pipeline = Pipeline(steps)

    # Add PCA components to hyperparameter search
    if use_pca:

        params = params.copy()

        params["pca__n_components"] = [
            2, 3, 4, 5, 6, 7, 8
        ]

    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_

    # Test prediction
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # Get the 5 CV fold scores
    best_index = grid.best_index_
    fold_scores = [grid.cv_results_[f"split{i}_test_score"][best_index]
        for i in range(5)
    ]

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_accuracy": accuracy,
        "test_f1": f1
    }

results = []

pca_model_scores = {}
for model_name, (model, params) in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("\nNO PCA")
    no_pca_result = run_experiment(model,params,use_pca=False)

    print("Best parameters:")
    print(no_pca_result["best_params"])
    print("Fold scores:",no_pca_result["fold_scores"])
    print("CV Mean:",round(no_pca_result["cv_mean"], 4))
    print("CV Std:",round(no_pca_result["cv_std"], 4))
    print("Test Accuracy:",round(no_pca_result["test_accuracy"], 4))
    print("Test F1:",round(no_pca_result["test_f1"], 4))


    print("\nWITH PCA")
    pca_result = run_experiment(
        model,
        params,
        use_pca=True
    )
    pca_model_scores[model_name] = pca_result["fold_scores"]
    print("Best parameters:")
    print(pca_result["best_params"])
    print("Fold scores:",pca_result["fold_scores"])
    print("CV Mean:",round(pca_result["cv_mean"], 4))
    print("CV Std:",round(pca_result["cv_std"], 4))
    print("Test Accuracy:",round(pca_result["test_accuracy"], 4))
    print("Test F1:",round(pca_result["test_f1"], 4))



    results.append({
        "Model": model_name,
        "No PCA CV Mean": no_pca_result["cv_mean"],
        "No PCA CV Std":no_pca_result["cv_std"],
        "No PCA Test Accuracy":no_pca_result["test_accuracy"],
        "No PCA Test F1":no_pca_result["test_f1"],
        "PCA CV Mean":pca_result["cv_mean"],
        "PCA CV Std":pca_result["cv_std"],
        "PCA Test Accuracy":pca_result["test_accuracy"],
        "PCA Test F1":pca_result["test_f1"],
        "Accuracy Change":pca_result["test_accuracy"] - no_pca_result["test_accuracy"],
        "F1 Change":pca_result["test_f1"] - no_pca_result["test_f1"],
        "Stability Change":pca_result["cv_std"] - no_pca_result["cv_std"]
    })


#comparison
results_df = pd.DataFrame(results)
print("\n\n")
print("=" * 100)
print("FINAL PCA vs NO-PCA COMPARISON")
print("=" * 100)

print(
    results_df[
        [
            "Model",
            "No PCA CV Mean",
            "PCA CV Mean",
            "No PCA Test Accuracy",
            "PCA Test Accuracy",
            "No PCA Test F1",
            "PCA Test F1",
            "Accuracy Change",
            "F1 Change",
            "Stability Change"
        ]
    ].round(4).to_string(index=False)
)
# Store PCA fold scores for ANOVA



from scipy.stats import f_oneway

# Get the 5-fold PCA scores for every classifier
anova_groups = list(pca_model_scores.values())

# Perform one-way ANOVA
f_stat, p_value = f_oneway(*anova_groups)

print("\n")
print("=" * 70)
print("ANOVA TEST - ALL CLASSIFIERS WITH PCA")
print("=" * 70)

print("F-statistic :", round(f_stat, 4))
print("p-value     :", round(p_value, 6))

alpha = 0.05

if p_value < alpha:
    print("\nResult: SIGNIFICANT")
    print("There is a statistically significant difference")
    print("between the classifiers after PCA.")
else:
    print("\nResult: NOT SIGNIFICANT")
    print("There is no statistically significant difference")
    print("between the classifiers after PCA.")


