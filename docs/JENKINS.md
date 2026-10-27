# CI/CD with Jenkins

The declarative pipeline in [`Jenkinsfile`](../Jenkinsfile) makes every deployment of the flight price service consistent and repeatable.

## Stages

| # | Stage | What happens | Fails the build when |
|---|---|---|---|
| 1 | Checkout | `checkout scm`, image tag = `<build>-<git sha>` | n/a |
| 2 | Setup Python | venv + pinned `requirements.txt` + `pip install -e .` | dependency conflict |
| 3 | Lint | `flake8` on src, api, streamlit_app, tests, scripts, dags | style or import errors |
| 4 | Unit tests | `pytest` (API, features, drift), JUnit report published | any test fails |
| 5 | Retrain models *(if `RETRAIN`)* | trains the 3 models and logs to MLflow | training error |
| 6 | **Model quality gate** | `scripts/check_model_quality.py --min-r2 0.95 --max-rmse 50` | the model is below the bar |
| 7 | Build Docker images | API + Streamlit, tagged with the build tag and `latest` | build error |
| 8 | **Smoke test container** | runs the image, waits for `/ready`, calls `/predict` | the container doesn't serve |
| 9 | Push images *(if `PUSH_IMAGE`)* | Docker Hub login from the `dockerhub-creds` credential | push error |
| 10 | Load into minikube *(otherwise)* | `minikube image load` | n/a |
| 11 | Deploy to Kubernetes *(if `DEPLOY`)* | `kubectl apply`, `set image`, change-cause annotation, `rollout status` | rollout doesn't become healthy in 180 s → **automatic `rollout undo`** |

Post actions archive `models/*.json` and `reports/*.json` so every build records which model it shipped.

## Parameters
`RETRAIN`, `PUSH_IMAGE`, `DEPLOY`, `REGISTRY` and `MODEL_RUN_ID` (filled in by the Airflow DAG when it triggers the job).

## Job setup
1. New item → **Pipeline** → *Pipeline script from SCM* → Git → repo URL → script path `Jenkinsfile`.
2. Add the credential `dockerhub-creds` (username/password) if images are pushed.
3. Triggers: **GitHub hook trigger** (on every push) and/or **Trigger builds remotely** (token), so the Airflow `deploy_new_model` task can call `JENKINS_DEPLOY_URL=http://<jenkins>/job/<job>/buildWithParameters?token=<token>`.

The Jenkins agent needs Python 3, Docker, kubectl and (for local clusters) minikube on its PATH.
