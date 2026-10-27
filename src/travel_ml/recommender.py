"""Hybrid hotel recommender.

Signals combined for an existing user:
  * own history   - how often the user stayed at each hotel (normalised)
  * user-based CF - what users with a similar booking pattern stay at
  * popularity    - small tie-breaker so new hotels/users still get a ranking

score = alpha * own + (1 - alpha) * similar_users + beta * popularity

Cold-start users (no bookings yet) are matched to existing users with the same
company and a similar age, and an optional nightly budget filters the catalog.

Evaluation is leave-last-out: each user's most recent hotel stay is hidden and
we check whether it appears in the top-k recommendations (HitRate@k, NDCG@k).

Usage:
    python -m travel_ml.recommender
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd

from travel_ml import config
from travel_ml.data import load_hotels, load_users, validate

logger = logging.getLogger(__name__)


class HotelRecommender:
    def __init__(self, alpha: float = 0.5, beta: float = 0.05, n_neighbours: int = 50):
        self.alpha = alpha
        self.beta = beta
        self.n_neighbours = n_neighbours

    # ------------------------------------------------------------------ fit
    def fit(self, hotels: pd.DataFrame, users: pd.DataFrame) -> "HotelRecommender":
        self.catalog_ = (hotels.groupby("name")
                         .agg(place=("place", "first"), price_per_night=("price", "mean"),
                              bookings=("travelCode", "count"), avg_days=("days", "mean"),
                              avg_total=("total", "mean"), guests=("userCode", "nunique"))
                         .sort_index())
        self.hotel_names_ = self.catalog_.index.tolist()

        counts = (hotels.pivot_table(index="userCode", columns="name", values="travelCode",
                                     aggfunc="count", fill_value=0)
                  .reindex(columns=self.hotel_names_, fill_value=0))
        self.user_index_ = {int(u): i for i, u in enumerate(counts.index)}
        raw = counts.to_numpy(dtype=float)
        self.user_pref_ = raw / raw.sum(axis=1, keepdims=True)
        # unit-normalised vectors: cosine similarity is then a single dot product per request,
        # which keeps the saved model small (no n_users x n_users matrix)
        vecs = np.log1p(raw)
        self.user_vecs_ = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        pop = self.catalog_["bookings"].to_numpy(dtype=float)
        self.popularity_ = pop / pop.sum()
        self.users_ = users.set_index("code")[["company", "age", "gender", "name"]]
        return self

    # -------------------------------------------------------------- scoring
    def _neighbour_scores(self, sims: np.ndarray) -> np.ndarray:
        top = np.argsort(-sims)[: self.n_neighbours]
        weights = sims[top]
        if weights.sum() <= 0:
            return self.popularity_.copy()
        return weights @ self.user_pref_[top] / weights.sum()

    def score_user(self, user_code: int) -> np.ndarray:
        if user_code not in self.user_index_:
            row = self.users_.loc[user_code] if user_code in self.users_.index else None
            if row is None:
                return self.popularity_.copy()
            return self.score_profile(row["age"], row["company"])
        i = self.user_index_[user_code]
        own = self.user_pref_[i]
        sims = self.user_vecs_ @ self.user_vecs_[i]
        sims[i] = 0  # ignore the user themself
        similar = self._neighbour_scores(sims)
        return self.alpha * own + (1 - self.alpha) * similar + self.beta * self.popularity_

    def score_profile(self, age: float, company: str | None = None) -> np.ndarray:
        """Cold start: average preferences of users from the same company with a similar age."""
        known = self.users_[self.users_.index.isin(self.user_index_.keys())]
        age_sim = np.exp(-((known["age"] - age) / 8.0) ** 2)
        if company:
            age_sim = age_sim * np.where(known["company"] == company, 1.0, 0.3)
        rows = [self.user_index_[int(c)] for c in known.index]
        sims = np.zeros(len(self.user_pref_))
        sims[rows] = age_sim.to_numpy()
        return self._neighbour_scores(sims) + self.beta * self.popularity_

    def _rank(self, scores, k, max_price=None, exclude=None) -> list[dict]:
        cat = self.catalog_.copy()
        cat["score"] = scores
        if max_price is not None:
            cat = cat[cat["price_per_night"] <= max_price]
        if exclude:
            cat = cat[~cat.index.isin(exclude)]
        cat = cat.sort_values("score", ascending=False).head(k)
        return [{"hotel": name, "place": r.place, "price_per_night": round(r.price_per_night, 2),
                 "avg_days": round(r.avg_days, 1), "score": round(float(r.score), 4)}
                for name, r in cat.iterrows()]

    def recommend(self, user_code: int, k: int = 3, max_price: float | None = None,
                  exclude_places: list[str] | None = None) -> list[dict]:
        exclude = None
        if exclude_places:
            exclude = self.catalog_.index[self.catalog_["place"].isin(exclude_places)].tolist()
        return self._rank(self.score_user(int(user_code)), k, max_price, exclude)

    def recommend_for_profile(self, age: float, company: str | None = None, k: int = 3,
                              max_price: float | None = None) -> list[dict]:
        return self._rank(self.score_profile(age, company), k, max_price)

    def user_history(self, user_code: int) -> dict:
        if int(user_code) not in self.user_index_:
            return {}
        prefs = self.user_pref_[self.user_index_[int(user_code)]]
        return {h: round(float(p), 3) for h, p in zip(self.hotel_names_, prefs) if p > 0}


# ---------------------------------------------------------------- evaluation
def leave_last_out_split(hotels: pd.DataFrame):
    ordered = hotels.sort_values(["userCode", "date", "travelCode"])
    test = ordered.groupby("userCode").tail(1)
    train = ordered.drop(test.index)
    test = test[test["userCode"].isin(train["userCode"])]
    return train, test


def evaluate(model: HotelRecommender, test: pd.DataFrame, k: int = 3) -> dict:
    hits, ndcg = [], []
    for row in test.itertuples():
        recs = [r["hotel"] for r in model.recommend(row.userCode, k=k)]
        if row.name in recs:
            hits.append(1)
            ndcg.append(1 / np.log2(recs.index(row.name) + 2))
        else:
            hits.append(0)
            ndcg.append(0)
    n_hotels = len(model.hotel_names_)
    return {f"hit_rate@{k}": float(np.mean(hits)), f"ndcg@{k}": float(np.mean(ndcg)),
            f"random_hit_rate@{k}": k / n_hotels, "n_test_users": len(test)}


def train(k: int = 3) -> dict:
    hotels, users = load_hotels(), load_users()
    validate(hotels, "hotels")
    # last stay per user = test, second-to-last = validation (used for tuning only)
    train_val_df, test_df = leave_last_out_split(hotels)
    train_df, val_df = leave_last_out_split(train_val_df)

    grid = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for beta in (0.0, 0.05, 0.2):
            for n_neighbours in (50, 200):
                m = HotelRecommender(alpha=alpha, beta=beta, n_neighbours=n_neighbours).fit(train_df, users)
                grid.append({"alpha": alpha, "beta": beta, "n_neighbours": n_neighbours,
                             **evaluate(m, val_df, k)})
    best = max(grid, key=lambda r: (r[f"hit_rate@{k}"], r[f"ndcg@{k}"]))
    params = {p: best[p] for p in ("alpha", "beta", "n_neighbours")}
    logger.info("best on validation: %s", best)

    # report on the untouched test stays, then refit on all bookings for serving
    test_metrics = evaluate(HotelRecommender(**params).fit(train_val_df, users), test_df, k)
    final = HotelRecommender(**params).fit(hotels, users)
    joblib.dump(final, config.RECOMMENDER_PATH, compress=3)
    summary = {"params": params, f"val_hit_rate@{k}": best[f"hit_rate@{k}"], **test_metrics}
    (config.REPORTS_DIR / "recommender_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # import through the package so the pickled class path is travel_ml.recommender, not __main__
    from travel_ml import recommender
    print(json.dumps(recommender.train(), indent=2))
