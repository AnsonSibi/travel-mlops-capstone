# Model tracking with MLflow

Tracking backend: SQLite (`mlflow.db`, which the model registry needs) with artefacts in `mlruns/`. Both are configurable through `MLFLOW_TRACKING_URI` / `MLFLOW_ARTIFACT_ROOT`.

```bash
make train          # creates/updates the experiments
make mlflow-ui      # http://localhost:5001
```

## What is logged

**Experiment `flight-price-regression`**: one run per candidate model (`ridge_baseline`, `decision_tree`, `random_forest`, `hist_gradient_boosting`):
* **params:** model name, all hyperparameters, train/test sizes, feature list
* **metrics:** `test_rmse`, `test_mae`, `test_r2`, `test_mape`, the same for train, `unseen_route_rmse` / `unseen_route_r2` (GroupKFold by route) and `train_seconds`
* **tags:** `model_family`, `dataset`, `trigger` (`manual` or `airflow`)
* **artefact:** the full sklearn pipeline with its **input signature** and an input example

**Experiment `gender-classification`**: GridSearchCV best params, accuracy, macro-F1, ROC-AUC, unseen-name accuracy, a classification report artefact and the model.

## Model registry and versioning

* After each training run the best candidate is registered as a new version of **`flight-price-regressor`**.
* The **`@champion`** alias moves to the new version **only if its test RMSE is ≤ the current champion's**. Otherwise the version gets **`@challenger`** and the serving artefact is left untouched.
* Each version is tagged with its `model_family`. `models/flight_price_model_meta.json` stores the MLflow `run_id` of the served model, and the API returns it as `model_version` in every response, so any prediction can be traced back to the exact run, parameters and data.

Load any version for comparison or rollback:
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
champion = mlflow.sklearn.load_model("models:/flight-price-regressor@champion")
v1 = mlflow.sklearn.load_model("models:/flight-price-regressor/1")
```
