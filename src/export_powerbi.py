import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
OUT = ROOT / "powerbi"
OUT.mkdir(exist_ok=True)

scores = pd.read_csv(APP / "scores.csv").sort_values("score", ascending=False).reset_index(drop=True)
dash = json.loads((APP / "dashboard.json").read_text())
info = json.loads((APP / "model_info.json").read_text())

# 1. customers: one row per cart user with rank, segment, and action
scores["rank"] = scores.index + 1
share = scores["rank"] / len(scores)
scores["segment"] = pd.cut(share, [0, 0.10, 0.30, 1.0], labels=["High intent", "Medium intent", "Low intent"])
scores["action"] = scores["segment"].map({"High intent": "Send reminder, no discount",
                                          "Medium intent": "Test small discount",
                                          "Low intent": "Don't contact"})
cols = ["user_id", "rank", "score", "segment", "action", "top_reasons", "view", "cart",
        "remove_from_cart", "sessions", "days_since_last", "days_since_last_cart"]
if "target" in scores.columns:
    cols.append("target")
scores[cols].rename(columns={"target": "bought_next_7_days"}).to_csv(OUT / "customers.csv", index=False)

# 2. gains curve: share of real buyers reached at each 1% of users contacted
if "target" in scores.columns:
    rows = []
    for p in range(1, 101):
        n = int(len(scores) * p / 100)
        reached = scores["target"].head(n).sum() / scores["target"].sum() * 100
        rows.append({"pct_contacted": p, "pct_buyers_reached": round(reached, 2), "random": p})
    pd.DataFrame(rows).to_csv(OUT / "gains.csv", index=False)

# 3. monthly story
m = pd.DataFrame(dash["monthly"])
m["buy_rate"] = (m["buyers"] / m["visitors"] * 100).round(2)
m.to_csv(OUT / "monthly.csv", index=False)

# 4. what drives the model
drivers = pd.Series(dash["drivers"], name="avg_shap").rename_axis("feature").reset_index()
drivers.sort_values("avg_shap", ascending=False).to_csv(OUT / "drivers.csv", index=False)

# 5. model results and order value
pd.DataFrame([{"pr_auc_february": info["february_pr_auc"],
               "random_pr_auc": info["february_random_pr_auc"],
               "avg_order_value": dash["avg_order_value"]}]).to_csv(OUT / "metrics.csv", index=False)

print("Saved to:", OUT)
for f in sorted(OUT.glob("*.csv")):
    print(" ", f.name)