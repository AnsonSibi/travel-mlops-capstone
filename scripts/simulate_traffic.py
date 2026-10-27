"""Send realistic requests to the API to exercise monitoring and autoscaling.

    python scripts/simulate_traffic.py --n 300                     # normal traffic
    python scripts/simulate_traffic.py --n 300 --drift             # skewed traffic -> drift alert
    python scripts/simulate_traffic.py --url http://$(minikube ip):30500 --n 2000 --workers 16

Normal mode samples real trips from flights.csv. Drift mode only sends long
first-class trips, which shifts the distance / flightType distributions.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from travel_ml import config


def build_payloads(n: int, drift: bool, seed: int = 7) -> list[dict]:
    flights = pd.read_csv(config.FLIGHTS_CSV)
    if drift:
        flights = flights[(flights["flightType"] == "firstClass") & (flights["distance"] > 700)]
    sample = flights.sample(n, replace=True, random_state=seed)
    cols = ["from", "to", "flightType", "agency", "date"]
    return sample[cols].to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    payloads = build_payloads(args.n, args.drift)
    session = requests.Session()

    def call(p):
        try:
            return session.post(f"{args.url}/predict", json=p, timeout=10).status_code
        except requests.RequestException:
            return "error"

    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(call, payloads))
    took = time.time() - start
    ok = sum(c == 200 for c in codes)
    print(f"sent {len(codes)} requests in {took:.1f}s ({len(codes) / took:.0f} req/s) - {ok} OK")
    if ok < len(codes):
        print("non-200 responses:", {c: codes.count(c) for c in set(codes) if c != 200})


if __name__ == "__main__":
    main()
