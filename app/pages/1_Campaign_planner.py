import streamlit as st

from charts import gains_fig
from common import kpi, load_dashboard, load_scores, section, setup

setup("Campaign planner", "📣", "Choose a budget and see the cost, the buyers you reach, and exactly who to contact.")
scores, dash = load_scores(), load_dashboard()
truth = "target" if "target" in scores.columns else "score"

# campaign settings
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Share of cart users to contact (%)", 5, 50, 10, 5)
    cost = c2.number_input("Cost per message", 0.0, 10.0, 0.05, 0.01)
    order_value = c3.number_input("Average order value", 1.0, 500.0, float(round(dash["avg_order_value"])), 1.0)

n = int(len(scores) * budget / 100)
picked = scores.head(n)
buyers = picked[truth].sum()

# what this campaign looks like
k = st.columns(4)
kpi(k[0], "Messages to send", f"{n:,}", f"{budget}% of {len(scores):,} cart users")
kpi(k[1], "Campaign cost", f"{n * cost:,.0f}", f"at {cost:.2f} per message")
kpi(k[2], "Buyers reached", f"{buyers:,.0f}", f"{buyers / scores[truth].sum():.0%} of all buyers", best=True)
kpi(k[3], "Order value at stake", f"{buyers * order_value:,.0f}", f"at {order_value:.0f} per order")

if truth == "target":
    section("Backtest", "What this list would have reached among real February buyers.")
    with st.container(border=True):
        st.plotly_chart(gains_fig(scores, budget))

# the list itself
section("Who to contact", "Ranked by chance of buying. Download it for the CRM or email tool.")
st.dataframe(
    picked[["user_id", "score", "segment", "top_reasons", "cart", "days_since_last"]],
    column_config={
        "user_id": "Customer ID",
        "score": st.column_config.ProgressColumn("Chance of buying", format="%.2f", min_value=0.0, max_value=1.0),
        "segment": "Segment",
        "top_reasons": "Why",
        "cart": "Cart items",
        "days_since_last": st.column_config.NumberColumn("Days since last visit", format="%.1f"),
    },
    hide_index=True, height=420,
)
st.download_button("⬇️ Download list for the CRM", picked.to_csv(index=False), "campaign_list.csv", "text/csv", type="primary")
st.caption("A reminder's real effect must be measured: keep a random 10% of the list uncontacted and compare how many buy.")