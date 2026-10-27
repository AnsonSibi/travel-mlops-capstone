# Streamlit web app

`streamlit_app/app.py` is the user-facing interface for the travel recommendation model and the travel insights.

```bash
make streamlit                                   # http://localhost:8501
API_URL=http://localhost:5000 make streamlit     # route flight price predictions through the REST API
```

## Tabs

| Tab | What you can do |
|---|---|
| 🏨 **Hotel recommendations** | Choose an existing user or a *new user* (age + company), set the number of suggestions and a max price per night, optionally *explore new places* (hide cities already visited). Shows ranked hotel cards, a score chart, and the user's booking history and KPIs. |
| 📊 **Travel insights** | KPIs (users, flights, stays, average fare) and interactive Plotly charts: popular destinations, monthly hotel revenue, price by class and agency, price vs distance, stays by company and gender. |
| 💸 **Flight price** | Pick route, class, agency and date and get a live prediction, from the Flask API when `API_URL` is set, otherwise from the local model. |
| 🧑 **Gender model** | Paste names and get predictions with confidence (`uncertain` below 60%). |
| 📈 **Monitoring** | Serving model metrics (test and unseen-route), recommender HitRate vs random, the distribution of live predictions and the latest drift report. |

Data and models are cached with `st.cache_data` / `st.cache_resource`, so the 270k-row flights table is loaded only once per session.
