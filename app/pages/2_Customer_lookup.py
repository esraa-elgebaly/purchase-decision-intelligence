import streamlit as st

from common import kpi, load_scores, section, setup

setup("Customer lookup", "👤", "Check any cart user: their chance of buying, why, and what to do this week.")
scores = load_scores()

# a searchable list: rank, customer id, and chance of buying
labels = [f"#{i + 1}  ·  {u}  ·  {s:.0%}" for i, (u, s) in enumerate(zip(scores["user_id"], scores["score"]))]
choice = st.selectbox("Pick a customer (click and type part of the ID, or #rank)", range(len(scores)),
                      format_func=lambda i: labels[i])

user = scores.iloc[choice]
rank = choice + 1
top_pct = rank / len(scores) * 100
in_campaign = top_pct <= 10

# the headline: chance, rank, and segment
c = st.columns(3)
kpi(c[0], "Chance of buying in 7 days", f"{user['score']:.0%}", f"{user['score'] / scores['score'].mean():.1f}x the average cart user")
kpi(c[1], "Rank this week", f"#{rank:,}", f"top {max(top_pct, 1):.0f}% of {len(scores):,} cart users")
kpi(c[2], "Segment", str(user["segment"]), "in this week's campaign" if in_campaign else "not in this week's campaign", best=in_campaign)

# a simple recommended action
st.write("")
if in_campaign:
    st.success("**Recommended:** send a cart reminder within the next day, without a discount.")
elif top_pct <= 30:
    st.info("**Recommended:** test a small discount, or keep in regular emails.")
else:
    st.warning("**Recommended:** don't contact this week.")

section("Why the model scored them like this")
reasons = [r for r in str(user["top_reasons"]).split("; ") if r and r != "nan"]
if reasons:
    st.markdown("".join(f'<span class="tag">{r.capitalize()}</span>' for r in reasons), unsafe_allow_html=True)
else:
    st.caption("No strong signal pushed this user toward buying.")

section("Last 14 days")
d = st.columns(4)
kpi(d[0], "Products viewed", f"{user['view']:.0f}")
kpi(d[1], "Added to cart", f"{user['cart']:.0f}")
kpi(d[2], "Removed from cart", f"{user['remove_from_cart']:.0f}")
kpi(d[3], "Days since last visit", f"{user['days_since_last']:.1f}")

st.write("")
cut10 = scores["score"].iloc[int(len(scores) * 0.10) - 1]
cut30 = scores["score"].iloc[int(len(scores) * 0.30) - 1]
st.caption(f"How the decision works: the top 10% (chance {cut10:.0%} or more) get a reminder, the next 20% "
           f"(chance {cut30:.0%} or more) may get a small offer, and everyone else is not contacted.")