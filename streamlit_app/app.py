"""Streamlit web app: hotel recommendations, travel insights and model playground.

Run:
    streamlit run streamlit_app/app.py

If API_URL is set (e.g. http://localhost:5000 or the Kubernetes service), flight
price predictions go through the Flask REST API; otherwise the app loads the
joblib model directly so it also works stand-alone (e.g. Streamlit Cloud).
"""
import json
import os

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from travel_ml import config
from travel_ml.data import load_flights, load_hotels, load_users
from travel_ml.monitoring import DRIFT_REPORT, load_prediction_log
from travel_ml.gender import predict_gender

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="Travel Recommender", page_icon="✈️", layout="wide")


# ----------------------------------------------------------- cached loaders
@st.cache_data(show_spinner="Loading travel data...")
def get_data():
    return load_flights(), load_hotels(), load_users()


@st.cache_resource
def get_models():
    models = {"recommender": joblib.load(config.RECOMMENDER_PATH)}
    for key, path in (("price", config.REGRESSION_MODEL_PATH), ("gender", config.GENDER_MODEL_PATH)):
        if path.exists():
            models[key] = joblib.load(path)
    if config.REGRESSION_META_PATH.exists():
        models["price_meta"] = json.loads(config.REGRESSION_META_PATH.read_text())
    return models


flights, hotels, users = get_data()
models = get_models()
rec = models["recommender"]
meta = models.get("price_meta", {})

st.title("✈️ Travel Recommendation & Insights")
st.caption("Hotel suggestions from booking history, travel trends, and live access to the "
           "flight price and gender models.")

tab_rec, tab_insights, tab_price, tab_gender, tab_monitor = st.tabs(
    ["🏨 Hotel recommendations", "📊 Travel insights", "💸 Flight price", "🧑 Gender model", "📈 Monitoring"])

# ------------------------------------------------------- recommendations
with tab_rec:
    mode = st.radio("Who are we recommending for?", ["Existing user", "New user (cold start)"],
                    horizontal=True)
    col_in, col_out = st.columns([1, 2])

    with col_in:
        k = st.slider("Number of suggestions", 1, 9, 3)
        max_price = st.slider("Max price per night", 50, 350, 350, step=10)
        if mode == "Existing user":
            labels = users.assign(label=users["code"].astype(str) + " - " + users["name"])
            choice = st.selectbox("User", labels["label"], index=0)
            user_code = int(choice.split(" - ")[0])
            profile = users.set_index("code").loc[user_code]
            st.markdown(f"**Company:** {profile.company}  \n**Age:** {profile.age}  \n"
                        f"**Gender:** {profile.gender}")
            explore = st.checkbox("Explore new places (hide cities already visited)")
            visited = hotels.loc[hotels["userCode"] == user_code, "place"].unique().tolist()
            recs = rec.recommend(user_code, k=k, max_price=max_price,
                                 exclude_places=visited if explore else None)
        else:
            age = st.number_input("Age", 18, 90, 30)
            company = st.selectbox("Company", sorted(users["company"].unique()))
            recs = rec.recommend_for_profile(age, company, k=k, max_price=max_price)
            user_code = None

    with col_out:
        if not recs:
            st.warning("No hotel matches these filters - try a higher budget.")
        else:
            st.subheader("Recommended hotels")
            cards = st.columns(min(len(recs), 3))
            for i, r in enumerate(recs):
                with cards[i % len(cards)]:
                    st.metric(f"#{i + 1} {r['hotel']}", r["place"], f"${r['price_per_night']:.0f}/night")
            rec_df = pd.DataFrame(recs)
            fig = px.bar(rec_df, x="score", y="hotel", orientation="h", color="place",
                         title="Recommendation score", labels={"score": "score", "hotel": ""})
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
            st.plotly_chart(fig, width="stretch")

        if user_code is not None:
            hist = hotels[hotels["userCode"] == user_code].sort_values("date")
            if len(hist):
                st.subheader(f"Booking history ({len(hist)} stays)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total spent on hotels", f"${hist['total'].sum():,.0f}")
                c2.metric("Avg nights per stay", f"{hist['days'].mean():.1f}")
                c3.metric("Cities visited", hist["place"].nunique())
                st.plotly_chart(px.histogram(hist, x="place", title="Stays per city"),
                                width="stretch")
            else:
                st.info("This user has no hotel bookings yet - recommendations come from similar users.")

# ------------------------------------------------------------- insights
with tab_insights:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{len(users):,}")
    c2.metric("Flights", f"{len(flights):,}")
    c3.metric("Hotel stays", f"{len(hotels):,}")
    c4.metric("Avg flight price", f"${flights['price'].mean():,.0f}")

    left, right = st.columns(2)
    with left:
        by_city = hotels.groupby("place").agg(stays=("travelCode", "count"),
                                              price=("price", "first")).reset_index()
        st.plotly_chart(px.bar(by_city.sort_values("stays"), x="stays", y="place", orientation="h",
                               color="price", title="Most popular destinations (hotel stays)"),
                        width="stretch")
        st.plotly_chart(px.box(flights.sample(20000, random_state=0), x="flightType", y="price",
                               color="agency", title="Flight price by class and agency"),
                        width="stretch")
    with right:
        monthly = (hotels.set_index("date").resample("ME")["total"].sum().reset_index())
        st.plotly_chart(px.line(monthly, x="date", y="total", title="Monthly hotel revenue"),
                        width="stretch")
        st.plotly_chart(px.scatter(flights.drop_duplicates(["from", "to", "flightType", "agency"]),
                                   x="distance", y="price", color="flightType",
                                   title="Price vs distance (each point = route/class/agency)"),
                        width="stretch")
    stay = hotels.merge(users, left_on="userCode", right_on="code", suffixes=("", "_user"))
    st.plotly_chart(px.histogram(stay, x="company", color="gender", barmode="group",
                                 title="Hotel stays by company and gender"), width="stretch")

# ---------------------------------------------------------- flight price
with tab_price:
    st.write("Predict a flight ticket price." + (f" Using the REST API at `{API_URL}`." if API_URL
                                                 else " Using the local model file."))
    allowed = meta.get("allowed_values", {})
    if not allowed:
        st.error("Flight price model metadata not found - train the model first.")
    else:
        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("From", allowed["from"])
        dest = c1.selectbox("To", [c for c in allowed["to"] if c != origin])
        ftype = c2.selectbox("Class", allowed["flightType"])
        agency = c2.selectbox("Agency", allowed["agency"])
        date = c3.date_input("Flight date")
        route = meta["routes"].get(f"{origin}|{dest}", {})
        c3.caption(f"Distance {route.get('distance', '?')} km · duration {route.get('time', '?')} h")

        if st.button("Predict price", type="primary"):
            payload = {"from": origin, "to": dest, "flightType": ftype, "agency": agency,
                       "date": date.strftime("%m/%d/%Y")}
            try:
                if API_URL:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    resp.raise_for_status()
                    price = resp.json()["predicted_price"]
                else:
                    row = pd.DataFrame([payload | route])
                    price = float(models["price"].predict(row)[0])
                st.success(f"Predicted price: **${price:,.2f}**")
            except requests.RequestException as exc:
                st.error(f"API call failed: {exc}")

# --------------------------------------------------------------- gender
with tab_gender:
    st.write("The gender model looks at character patterns in the first name. If it is less than "
             "60% sure it answers *uncertain*.")
    names = st.text_area("Names (one per line)", "Maria Silva\nJohn Smith\nAlex Morgan")
    if "gender" in models and st.button("Classify"):
        preds = predict_gender(models["gender"], [n for n in names.splitlines() if n.strip()])
        st.dataframe(pd.DataFrame([{"name": p["name"], "prediction": p["gender"],
                                    "confidence": p["confidence"]} for p in preds]),
                     width="stretch")

# ------------------------------------------------------------ monitoring
with tab_monitor:
    st.subheader("Serving model")
    if meta:
        m = meta["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", meta["model_name"])
        c2.metric("Test RMSE", f"{m['rmse']:.2f}")
        c3.metric("Test R²", f"{m['r2']:.4f}")
        c4.metric("Unseen-route R²", f"{m.get('unseen_route_r2', float('nan')):.3f}")
        st.caption(f"MLflow run `{meta['mlflow_run_id']}` · trained {meta['trained_at']}")

    rec_metrics = config.REPORTS_DIR / "recommender_metrics.json"
    if rec_metrics.exists():
        rm = json.loads(rec_metrics.read_text())
        st.subheader("Recommender (leave-last-out)")
        c1, c2, c3 = st.columns(3)
        c1.metric("HitRate@3", f"{rm['hit_rate@3']:.3f}")
        c2.metric("Random baseline", f"{rm['random_hit_rate@3']:.3f}")
        c3.metric("NDCG@3", f"{rm['ndcg@3']:.3f}")

    st.subheader("Live predictions & drift")
    log = load_prediction_log()
    if log.empty:
        st.info("No predictions logged yet - call the /predict endpoint to populate this view.")
    else:
        st.metric("Logged predictions", len(log))
        st.plotly_chart(px.histogram(log, x="prediction", nbins=30, title="Predicted price distribution"),
                        width="stretch")
    if DRIFT_REPORT.exists():
        st.json(json.loads(DRIFT_REPORT.read_text()))
