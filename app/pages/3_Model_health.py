import numpy as np
import pandas as pd
import streamlit as st

from charts import drift_fig
from common import kpi, load_info, load_scores, section, setup

setup("Model health", "🩺", "How good the model is, and whether it still fits new data.")
info, scores = load_info(), load_scores()

# how good the model was on a month it never saw
pr, rnd = info["february_pr_auc"], info["february_random_pr_auc"]
k = st.columns(4)
kpi(k[0], "PR-AUC on February", f"{pr:.3f}", "a month the model never saw")
kpi(k[1], "Random guess", f"{rnd:.3f}", "the buy rate of cart users")
kpi(k[2], "Times better than random", f"{pr / rnd:.1f}x", "", best=True)
kpi(k[3], "Trees in the model", f"{info['number_of_trees']}", "LightGBM, trained on " + ", ".join(info["trained_on"]))

with st.expander("Model settings"):
    st.json(info["settings"])


def psi(expected, actual, bins=10):
    # how much a feature's shape changed compared with the reference week
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True, sort=False).clip(lower=1e-4)
    a = pd.cut(actual.clip(edges[0], edges[-1]), edges, include_lowest=True).value_counts(normalize=True, sort=False).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


section("Data drift check", "Upload a newer scores file from the pipeline. Below 0.1 is stable, "
        "0.1 to 0.25 is worth watching, above 0.25 may need retraining.")
file = st.file_uploader("Scores file (CSV from src/score.py)", type="csv")
if file is not None:
    new = pd.read_csv(file)
    drift = pd.Series({f: psi(scores[f], new[f]) for f in info["features"] if f in new.columns}).sort_values()
    with st.container(border=True):
        st.plotly_chart(drift_fig(drift))
    changed = list(drift[drift > 0.25].index)
    if changed:
        st.warning(f"Big change in: {', '.join(changed)}. Consider retraining on newer data.")
    else:
        st.success("No big changes. The model should still fit this data.")