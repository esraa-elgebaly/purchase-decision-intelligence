import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
APP = ROOT / "app"

scores = pd.read_csv(APP / "scores.csv")
booster = lgb.Booster(model_file=str(APP / "model.txt"))
feats = json.loads((APP / "model_info.json").read_text())["features"]

# what drives the model: average shap size on this week's users
contrib = booster.predict(scores[feats], pred_contrib=True)[:, :-1]
drivers = dict(zip(feats, np.abs(contrib).mean(axis=0).round(4).tolist()))

# monthly sales story, one month at a time
monthly = []
for m in ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan", "2020-Feb"]:
    print("Reading", m)
    d = pd.read_csv(DATA / f"{m}.csv", usecols=["event_time", "event_type", "product_id", "price", "user_id", "user_session"])
    d = d.drop_duplicates().dropna(subset=["user_session"])
    b = d[(d["event_type"] == "purchase") & (d["price"] > 0)]
    monthly.append({"month": m, "visitors": int(d["user_id"].nunique()), "buyers": int(b["user_id"].nunique()),
                    "sales": round(float(b["price"].sum()), 2),
                    "spend_per_buyer": round(float(b.groupby("user_id")["price"].sum().mean()), 2)})

dashboard = {"drivers": drivers, "monthly": monthly, "avg_order_value": monthly[-1]["spend_per_buyer"]}
(APP / "dashboard.json").write_text(json.dumps(dashboard, indent=2))
print("Saved:", APP / "dashboard.json")