import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

from features import clean_month, build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "app"

# a plain-language reason for each feature, filled with the user's value
REASONS = {
    "days_since_last": "active {:.1f} days ago",
    "days_since_last_cart": "last cart action {:.1f} days ago",
    "cart": "added {:.0f} items to cart",
    "view": "viewed {:.0f} products",
    "remove_from_cart": "edited cart ({:.0f} items removed)",
    "sessions": "visited {:.0f} times",
    "past_purchases": "bought {:.0f} items recently",
    "days_since_first": "first seen {:.1f} days ago",
    "views_per_session": "{:.1f} products per visit",
    "remove_per_cart": "{:.1f} removes per cart item",
    "avg_brand_rate": "likes brands that sell well ({:.0%})",
    "avg_price": "average price seen {:.2f}",
    "last_hour": "last visit at {:.0f}:00 UTC",
    "last_weekday": "last visit on day {:.0f} of the week",
}


def main():
    parser = argparse.ArgumentParser(description="Score cart users: who is likely to buy in the next 7 days?")
    parser.add_argument("--data", required=True, help="raw events CSV for the month")
    parser.add_argument("--cutoff", required=True, help="scoring date, for example 2020-02-15")
    parser.add_argument("--budget", type=float, default=10, help="share of users to contact, in percent")
    args = parser.parse_args()

    # load the model and the list of features it expects
    model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    features = info["features"]

    # clean, build features, keep users with cart activity
    print("Cleaning and building features...")
    users = build_features(clean_month(args.data), args.cutoff).reset_index()
    users = users[users["cart"] > 0].copy()

    # score, rank, and flag the users to contact
    users["score"] = model.predict(users[features])
    users = users.sort_values("score", ascending=False).reset_index(drop=True)
    n = int(len(users) * args.budget / 100)
    users["contact"] = users.index < n

    # the two biggest reasons for each user, from the shap values
    contrib = model.predict(users[features], pred_contrib=True)[:, :-1]
    reasons = []
    for row, values in zip(users[features].itertuples(index=False), contrib):
        best = np.argsort(-values)[:2]
        reasons.append("; ".join(REASONS[features[j]].format(row[j]) for j in best if values[j] > 0))
    users["top_reasons"] = reasons

    # keep the real outcome only if the file has the week after the cutoff
    cols = ["user_id", "score", "contact", "top_reasons"] + features
    has_outcome = users["target"].sum() > 0
    if has_outcome:
        cols.append("target")

    # save the list for the CRM or email tool
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"scores_{args.cutoff}.csv"
    users[cols].to_csv(out_file, index=False)

    print(f"Scored {len(users):,} cart users | {n:,} to contact ({args.budget:.0f}%)")
    print(f"Saved: {out_file}")

    # check how the list would have done
    if has_outcome:
        reached = users.loc[users["contact"], "target"].sum() / users["target"].sum()
        pr_auc = average_precision_score(users["target"], users["score"])
        print(f"Backtest | PR-AUC: {pr_auc:.3f} (random {users['target'].mean():.3f}) | buyers reached by the list: {reached:.0%}")


if __name__ == "__main__":
    main()