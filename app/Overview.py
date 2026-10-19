import streamlit as st

from charts import drivers_fig, gains_fig, monthly_fig, segments_fig
from common import card, kpi, load_dashboard, load_info, load_scores, section, setup, TEAL, AMBER, LIGHT

setup("Cart Recovery Intelligence", "🛒",
      "70% of cart users never buy. This tool ranks them by their chance of buying in the next 7 days, "
      "so the store contacts fewer people and still reaches most buyers.",
      ["Week of 2020-02-15", "LightGBM · tested on unseen February", "Updated weekly by the pipeline"])
scores, info, dash = load_scores(), load_info(), load_dashboard()
aov = dash["avg_order_value"]
truth = "target" if "target" in scores.columns else "score"
high = scores[scores["segment"] == "High intent"]

# 1. the situation
k = st.columns(5)
kpi(k[0], "Cart users this week", f"{len(scores):,}", "active cart, last 14 days")
kpi(k[1], "Buyers in the next 7 days", f"{scores[truth].sum():,.0f}", f"only {scores[truth].mean():.1%} of cart users")
kpi(k[2], "High-intent users", f"{len(high):,}", "top 10% by model score")
kpi(k[3], "Revenue at stake", f"{high['score'].sum() * aov:,.0f}", f"top 10% · {aov:.0f} per order")
kpi(k[4], "Model vs random", f"{info['february_pr_auc'] / info['february_random_pr_auc']:.1f}x", "on unseen February data")

# 2. the decision: same budget, very different results
section("This week's decision", "Same number of messages, very different results. Buyers are real February buyers (backtest).")
n = int(len(scores) * 0.10)
total = scores[truth].sum()
options = [("Contact everyone", len(scores), total, False),
           ("Random 10% of users", n, total * 0.10, False),
           ("Top 10% by the model ✅", n, scores[truth].head(n).sum(), True)]
for col, (name, msgs, buyers, best) in zip(st.columns(3), options):
    kpi(col, name, f"{buyers:,.0f} buyers", f"{msgs:,} messages · 1 in {msgs / buyers:.0f} buys", best)

# 3. what to do with each group
section("Action plan by segment", "Each group gets a different action, so money goes where it can change a decision.")
seg = scores.groupby("segment", observed=True).agg(users=("score", "size"), buyers=(truth, "sum"))
plan = [("High intent", TEAL, "Send a reminder, no discount", "Likely to buy anyway, so a discount would mostly cost margin."),
        ("Medium intent", AMBER, "Test a small discount", "Interested but undecided: where an offer may change minds."),
        ("Low intent", "#64748B", "Don't contact this week", "Saves budget and avoids annoying customers.")]
for col, (name, color, action, why) in zip(st.columns(3), plan):
    meta = f"{seg.loc[name, 'users']:,} users · {seg.loc[name, 'buyers']:,.0f} buyers"
    card(col, name, color, meta, action, why)

# 4. the evidence
section("The evidence", "Why the ranking can be trusted, and what it looks at.")
charts = [("The ranking finds buyers early", gains_fig(scores) if truth == "target" else None),
          ("A small group holds many buyers", segments_fig(scores)),
          ("Sales follow the buy rate, not traffic", monthly_fig(dash["monthly"])),
          ("Recent activity signals buying intent (SHAP)", drivers_fig(dash["drivers"]))]
charts = [c for c in charts if c[1] is not None]
for i in range(0, len(charts), 2):
    for col, (title, fig) in zip(st.columns(2), charts[i:i + 2]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.plotly_chart(fig, key=title)

# 5. rules the data gave us
section("Rules the data gave us")
rules = [("Act within a day", "Half of buyers check out within an hour of their first cart."),
         ("Don't wait a week", "After a week without activity, the chance of buying drops sharply."),
         ("Prove it with a holdout", "Keep 10% of the list uncontacted and compare how many buy.")]
for col, (name, text) in zip(st.columns(3), rules):
    card(col, name, TEAL, "", "", text)