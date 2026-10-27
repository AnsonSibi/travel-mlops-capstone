# Video presentation outline (regression model, ~40 minutes)

The guideline sections are kept in order; each has a time budget, talking points and what to show on screen.

| # | Section | Time | Show on screen |
|---|---|---|---|
| 1 | Introduction | 2 min | README architecture diagram |
| 2 | Problem understanding | 3 min | README "Motivation" |
| 3 | Data understanding & use-case mapping | 4 min | Notebook 1 sections 1–3, data relationship diagram |
| 4 | EDA & hypothesis testing | 5 min | Notebook 1 charts 3, 5, 6, 9, 14 + hypotheses |
| 5 | Feature engineering & model development | 6 min | Notebook 1 sections 6–7, comparison tables |
| 6 | REST API & serving design | 4 min | `api/app.py`, Postman/curl demo |
| 7 | Docker | 3 min | Dockerfile, `docker build`, `docker ps` |
| 8 | Kubernetes & scaling | 4 min | manifests, `kubectl get pods,hpa`, load test |
| 9 | Airflow scheduling | 4 min | DAG graph views, task logs, drift-triggered run |
| 10 | Jenkins CI/CD | 3 min | Jenkinsfile, stage view, rollback |
| 11 | MLflow tracking | 3 min | experiment comparison, registry aliases |
| 12 | Recommendation layer, Streamlit & monitoring | 3 min | Streamlit tabs, `/metrics`, drift report |
| 13 | Reliability, challenges & optimisation | 2 min | n/a |
| 14 | Learnings & future work | 2 min | README "Future improvements" |

## Talking points

**1. Introduction:** an end-to-end ML *system* for a travel platform with three models (flight price regression, gender classification, hotel recommendation). Stack: Python, scikit-learn, Flask, Docker, Kubernetes, Airflow, Jenkins, MLflow, Streamlit. The video focuses on the regression model's path to production.

**2. Problem:** quoting accurate prices in real time; wrong prices lose bookings or margin. ML learns the pricing logic from 270k flights. A notebook model can't serve traffic, scale, retrain or be audited, which is why it needs productionising.

**3. Data:** users (1,340), flights (271,888 legs, 2 per trip), hotels (40,552). `userCode` links users to flights and hotels, `travelCode` links a trip's flights to its hotel. Flights → regression, users → classification, hotels (+ users) → recommendation.

**4. EDA:** no missing values, duplicates or outliers. Price = route + class + agency tariff (490 combinations, one price each). Price is roughly linear in distance with a per-class slope. No seasonality, and the weekend effect comes from the route mix (Hypothesis 3).

**5. Modelling:** one-hot encoding, date features, distance × class interactions. Ridge vs Random Forest vs Hist Gradient Boosting with CV tuning. On known routes the trees are exact; on unseen routes (GroupKFold) Ridge generalises best. I chose tuned HGB for production and kept the unseen-route metric as a robustness check. Explainability: permutation importance (class > distance > route > agency; date ≈ 0).

**6. API:** request → validation → route lookup → same sklearn pipeline as training → JSON response with `model_version` → prediction log + metrics. Health and readiness endpoints for orchestration.

**7. Docker:** slim base, pinned versions (same sklearn as training), cached layers, non-root user, healthcheck, gunicorn.

**8. Kubernetes:** 2 replicas, probes, resource limits, rolling update with zero downtime, HPA 2–6 pods at 60% CPU. Live demo: load test → HPA scales up.

**9. Airflow:** weekly training DAG (validate → gate → features → train/register → branch → deploy or keep) and daily drift DAG (PSI > 0.2 → trigger retraining). Show the drifted-traffic run.

**10. Jenkins:** lint → tests → (retrain) → quality gate → build → smoke test → push → deploy → automatic rollback. Triggered by git push and by Airflow after a new champion.

**11. MLflow:** runs with params, metrics and signatures; registry with `@champion` / `@challenger`; the API returns the run id for traceability.

**12. Recommendation & monitoring:** the hybrid recommender (similar users + popularity, tuned on validation stays), Streamlit tabs, Prometheus metrics, drift report.

**13. Challenges:** joining three datasets correctly, a pickling issue with custom pipeline steps (fixed by putting them in an importable package), keeping training and serving features identical, MLflow serialization formats, Airflow's runtime adding a dill dependency to pickles.

**14. Learnings:** a model is ~10% of a production ML system; validation, versioning, reproducibility and monitoring are the rest. Future work: dynamic pricing features, auth and rate limiting, a cloud cluster with a remote MLflow server, Grafana dashboards, canary deployments.

## Follow-up interview questions (short answers)

* **Productionising real-time travel pricing?** Package the pipeline (not just the estimator) so features match; serve it behind a stateless API with validation; containerise; autoscale on Kubernetes; log predictions; monitor drift and latency; retrain on a schedule or when drift is detected; promote through a registry with a quality gate.
* **Why Docker and Kubernetes?** Docker gives one reproducible artefact with pinned dependencies. Kubernetes adds replicas, self-healing, load balancing, autoscaling and zero-downtime rollouts and rollbacks.
* **How does MLflow help with versions?** Every run records params, metrics and the model. The registry versions models, and aliases (`@champion`) decouple "which model is in production" from the code, so promotion and rollback are a metadata change.
* **Retraining in production?** Scheduled (weekly Airflow DAG) plus event-driven (drift DAG triggers it); a quality gate and champion comparison prevent regressions; CI/CD redeploys only promoted models.
* **Scaling for a large platform?** Horizontal autoscaling of stateless API pods, batch endpoints, caching of frequent routes, a remote MLflow/feature store, async logging to a queue (e.g. Kafka) for monitoring, and multi-region clusters.
