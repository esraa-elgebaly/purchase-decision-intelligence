import pandas as pd
import plotly.graph_objects as go

from common import TEAL, CORAL, AMBER, DARK, LIGHT

LAYOUT = dict(height=320, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=13))


def segments_fig(scores):
    # each group's share of users next to its share of buyers
    truth = "target" if "target" in scores.columns else "score"
    seg = scores.groupby("segment", observed=True).agg(users=("score", "size"), buyers=(truth, "sum"))
    seg = seg / seg.sum() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(y=seg.index, x=seg["users"], name="Share of cart users", orientation="h", marker_color=LIGHT,
                         text=[f"{v:.0f}%" for v in seg["users"]]))
    fig.add_trace(go.Bar(y=seg.index, x=seg["buyers"], name="Share of buyers", orientation="h", marker_color=TEAL,
                         text=[f"{v:.0f}%" for v in seg["buyers"]]))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.12), **LAYOUT)
    return fig


def gains_fig(scores, budget=10):
    # share of real buyers reached when contacting the top users
    reached = scores["target"].cumsum() / scores["target"].sum() * 100
    pct = (scores.index + 1) / len(scores) * 100
    at = reached.iloc[max(int(len(scores) * budget / 100) - 1, 0)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct[::50], y=reached.iloc[::50], line=dict(color=TEAL, width=3), name="Model",
                             fill="tozeroy", fillcolor="rgba(15,118,110,0.08)"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], line=dict(color=LIGHT, dash="dash"), name="Random"))
    fig.add_annotation(x=budget, y=at, text=f"<b>Top {budget}% → {at:.0f}% of buyers</b>", showarrow=True,
                       arrowcolor=DARK, ax=90, ay=55, bgcolor="white")
    fig.update_layout(showlegend=False, xaxis_title="% of cart users contacted", yaxis_title="% of buyers reached", **LAYOUT)
    return fig


def monthly_fig(monthly):
    # sales per month with the buy rate on a second axis
    m = pd.DataFrame(monthly)
    m["buy_rate"] = m["buyers"] / m["visitors"] * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["month"], y=m["sales"], name="Sales", marker_color=[LIGHT, TEAL, CORAL, LIGHT, LIGHT]))
    fig.add_trace(go.Scatter(x=m["month"], y=m["buy_rate"], name="Buy rate (%)", yaxis="y2",
                             line=dict(color=DARK, width=3), mode="lines+markers"))
    fig.update_layout(legend=dict(orientation="h", y=1.12),
                      yaxis2=dict(overlaying="y", side="right", title="Buy rate (%)", range=[0, m["buy_rate"].max() * 1.3]), **LAYOUT)
    return fig


def drivers_fig(drivers):
    # average shap effect of each feature, top 4 highlighted
    d = pd.Series(drivers).sort_values()
    fig = go.Figure(go.Bar(x=d.values, y=[n.replace("_", " ") for n in d.index], orientation="h",
                           marker_color=[TEAL if v >= d.iloc[-4] else LIGHT for v in d.values]))
    fig.update_layout(xaxis_title="Average effect on the score", **LAYOUT)
    return fig


def drift_fig(drift):
    # psi per feature: green stable, amber watch, coral retrain
    colors = [CORAL if v > 0.25 else (AMBER if v > 0.1 else TEAL) for v in drift.values]
    fig = go.Figure(go.Bar(x=drift.values, y=list(drift.index), orientation="h", marker_color=colors))
    fig.add_vline(x=0.1, line_dash="dot", line_color=DARK)
    fig.add_vline(x=0.25, line_dash="dash", line_color=DARK)
    fig.update_layout(xaxis_title="PSI (change compared with this week)", **dict(LAYOUT, height=420))
    return fig