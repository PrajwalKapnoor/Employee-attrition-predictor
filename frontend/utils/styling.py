"""
styling.py
==========
Small shared helpers so every page uses the same risk-tier colors and
the same "metric card" look, instead of each page reinventing CSS.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

TIER_COLORS = {
    "High Risk": "#ef5350",
    "Medium Risk": "#ffca28",
    "Low Risk": "#66bb6a",
}
TIER_EMOJI = {
    "High Risk": "🔴",
    "Medium Risk": "🟡",
    "Low Risk": "🟢",
}


@st.cache_data
def load_asset(filename: str) -> dict:
    """Loads a precomputed JSON asset once and caches it across reruns.
    All EDA / business-metrics data lives here — nothing is recomputed
    live in the frontend."""
    with open(ASSETS_DIR / filename) as f:
        return json.load(f)


def render_figure(fig_dict: dict, key: str | None = None):
    """Rebuilds a Plotly figure from its saved JSON dict and renders it.

    The saved dict deliberately has no embedded theme (see
    scripts/export_artifacts.py's fig_to_dict for why) — so the dark theme
    is re-applied here, by name, using whatever Plotly version is actually
    installed. This keeps charts visually consistent without ever
    depending on Plotly's internal template schema staying identical
    across versions.
    """
    fig = go.Figure(fig_dict)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, key=key)


def risk_badge(tier: str) -> str:
    """Returns a colored markdown badge for a risk tier, e.g. '🔴 High Risk'."""
    return f"{TIER_EMOJI.get(tier, '')} **{tier}**"


def inject_base_css():
    """A few small CSS tweaks that Streamlit's theme config can't express:
    tighter card borders and consistent metric spacing. Kept minimal on
    purpose — the .streamlit/config.toml theme does the heavy lifting."""
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 12px 16px;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #2a2a2a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )