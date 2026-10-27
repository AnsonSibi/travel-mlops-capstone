"""Gender classification from the user's first name.

The users table has three labels: male, female and none. "none" means the user
did not declare a gender, and the same first names appear under "none" and
under male/female, so it is not something a model can (or should) predict.
We therefore train on male/female only and return "uncertain" when the model's
confidence is below a threshold.

Features: character n-grams of the first name (e.g. names ending in "a" are
mostly female). Model: Logistic Regression tuned with GridSearchCV.

Usage:
    python -m travel_ml.train_classifier
"""
import json
import logging

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from travel_ml import config
from travel_ml.data import load_users, validate
from travel_ml.gender import UNCERTAIN_THRESHOLD, first_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_classifier")

EXPERIMENT = "gender-classification"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("first_name", FunctionTransformer(first_name)),
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def train() -> dict:
    users = load_users()
    validate(users, "users")
    labelled = users[users["gender"].isin(["male", "female"])].reset_index(drop=True)
    X, y = labelled["name"], labelled["gender"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)

    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"tfidf__ngram_range": [(1, 3), (2, 4), (1, 5)], "clf__C": [0.5, 1, 5, 10]},
        cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1_macro", n_jobs=-1,
    )
    with mlflow.start_run(run_name="char_ngram_logreg"):
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, list(model.classes_).index("male")]
        metrics = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_f1_macro": f1_score(y_test, pred, average="macro"),
            "test_roc_auc": roc_auc_score((y_test == "male").astype(int), proba),
            "cv_best_f1_macro": grid.best_score_,
        }
        # How well do we do on first names never seen in training?
        groups = first_name(X).values
        metrics["unseen_name_accuracy"] = float(cross_val_score(
            build_pipeline().set_params(**grid.best_params_), X, y, cv=GroupKFold(5), groups=groups).mean())

        mlflow.log_params({k.replace("__", "_"): str(v) for k, v in grid.best_params_.items()})
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test),
                           "uncertain_threshold": UNCERTAIN_THRESHOLD})
        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_test, pred), "classification_report.txt")
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                                 registered_model_name="gender-classifier")

    joblib.dump(model, config.GENDER_MODEL_PATH, compress=3)
    logger.info("best params %s | %s", grid.best_params_, metrics)
    return {"best_params": {k: str(v) for k, v in grid.best_params_.items()}, **metrics}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
