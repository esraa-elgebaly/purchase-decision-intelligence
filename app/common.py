import json
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

HERE = Path(__file__).parent
TEAL, CORAL, AMBER, DARK, LIGHT = "#0F766E", "#E76F51", "#F4A261", "#1F2937", "#94A3B8"

# one clean look for every plotly chart
pio.templates.default = "simple_white"


def setup(title, icon, text, chips=()):
    # page settings, the shared style, and a banner at the top
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(f"<style>{(HERE / 'style.css').read_text()}</style>", unsafe_allow_html=True)
    tags = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><div class="title">{icon} {title}</div>'
                f'<div class="text">{text}</div>{tags}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, label, value, note="", best=False):
    # a card with a small label, a big number, and a short note
    col.markdown(f'<div class="kpi{" best" if best else ""}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def card(col, name, color, meta, action, why):
    # a card with a colored edge: who, what to do, and why
    col.markdown(f'<div class="card" style="--c:{color}"><div class="name">{name}</div>'
                 f'<div class="meta">{meta}</div><div class="act">{action}</div>'
                 f'<div class="why">{why}</div></div>', unsafe_allow_html=True)


@st.cache_data
def load_scores():
    # the ranked list from the weekly pipeline, split into three intent groups
    s = pd.read_csv(HERE / "scores.csv").sort_values("score", ascending=False).reset_index(drop=True)
    share = (s.index + 1) / len(s)
    s["segment"] = pd.cut(share, [0, 0.10, 0.30, 1.0], labels=["High intent", "Medium intent", "Low intent"])
    return s


@st.cache_data
def load_info():
    return json.loads((HERE / "model_info.json").read_text())


@st.cache_data
def load_dashboard():
    return json.loads((HERE / "dashboard.json").read_text())