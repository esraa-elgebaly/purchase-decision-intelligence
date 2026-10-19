import pandas as pd


def clean_month(path):
    # same cleaning decisions as the notebook (Step 3)
    d = pd.read_csv(path)
    d["event_time"] = pd.to_datetime(d["event_time"].str.replace(" UTC", ""))
    d = d.drop_duplicates()
    d = d.drop(columns=["category_code"], errors="ignore")
    d["brand"] = d["brand"].fillna("unknown")
    d = d.dropna(subset=["user_session"])
    d.loc[d["price"] <= 0, "price"] = None
    return d


def build_features(d, cutoff, history_days=14, target_days=7):
    cutoff = pd.Timestamp(cutoff)

    # the 14 days before the cutoff, and the 7 days after it
    past = d[(d["event_time"] >= cutoff - pd.Timedelta(days=history_days)) & (d["event_time"] < cutoff)]
    future = d[(d["event_time"] >= cutoff) & (d["event_time"] < cutoff + pd.Timedelta(days=target_days))]

    # count each action per user
    feats = past.pivot_table(index="user_id", columns="event_type", values="product_id", aggfunc="count", fill_value=0)
    feats = feats.rename(columns={"purchase": "past_purchases"})
    feats.columns.name = None

    # visits and timing
    first_time = past.groupby("user_id")["event_time"].min()
    last_time = past.groupby("user_id")["event_time"].max()
    last_cart = past[past["event_type"] == "cart"].groupby("user_id")["event_time"].max()
    feats["sessions"] = past.groupby("user_id")["user_session"].nunique()
    feats["days_since_last"] = (cutoff - last_time).dt.total_seconds() / 86400
    feats["days_since_first"] = (cutoff - first_time).dt.total_seconds() / 86400

    # users with no cart get the full window length
    feats["days_since_last_cart"] = ((cutoff - last_cart).dt.total_seconds() / 86400).reindex(feats.index).fillna(history_days)
    feats["avg_price"] = past.groupby("user_id")["price"].mean()
    feats["last_hour"] = last_time.dt.hour
    feats["last_weekday"] = last_time.dt.dayofweek

    # brand buy rate from the history only, then the average for each user
    views = past[past["event_type"] == "view"]
    brand_rate = (past[past["event_type"] == "purchase"].groupby("brand")["user_id"].nunique() / views.groupby("brand")["user_id"].nunique()).fillna(0)
    seen = views[["user_id", "brand"]].drop_duplicates()
    seen["brand_rate"] = seen["brand"].map(brand_rate)
    feats["avg_brand_rate"] = seen.groupby("user_id")["brand_rate"].mean()

    # ratios
    feats["views_per_session"] = feats["view"] / feats["sessions"]
    feats["remove_per_cart"] = feats["remove_from_cart"] / feats["cart"].replace(0, 1)

    # target: did the user buy in the next 7 days? (all 0 if the future isn't in the file yet)
    buyers = future[future["event_type"] == "purchase"]["user_id"].unique()
    feats["target"] = feats.index.isin(buyers).astype(int)
    return feats