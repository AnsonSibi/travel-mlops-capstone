# Docker

| Image | Dockerfile | Port | Contents |
|---|---|---|---|
| `travel-ml-api` | `docker/Dockerfile.api` | 5000 | `requirements-api.txt`, `travel_ml` package, `api/`, `models/`, served by gunicorn |
| `travel-ml-streamlit` | `docker/Dockerfile.streamlit` | 8501 | Streamlit deps, `travel_ml`, app, models, raw data |

Design choices:
* **`python:3.12-slim`** base with pinned dependencies, so builds are reproducible and the same scikit-learn version that trained the model also loads it.
* **Layer caching:** requirements are installed before the code is copied, so code-only changes rebuild in seconds.
* **Non-root user** (`appuser`, uid 1000), with only `logs/` and `reports/` writable.
* **HEALTHCHECK** on `/health` (API) and `/_stcore/health` (Streamlit).
* **`.dockerignore`** keeps `.venv`, `mlruns`, notebooks and tests out of the build context.

```bash
docker build -f docker/Dockerfile.api -t travel-ml-api:latest .
docker run -d -p 5000:5000 --name travel-api travel-ml-api:latest
curl localhost:5000/ready

docker compose up --build          # API + Streamlit (Streamlit calls the API via http://api:5000)
```
