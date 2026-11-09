"""
Fairness AI - Central Theme Module (app/theme.py)
=================================================
Provides centralized design tokens, Plotly styling templates, and CSS injection
for the candidate scoring and algorithmic governance dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------------------------------------------------------
# 1. CENTRAL COLOR PALETTE
# -----------------------------------------------------------------------------
COLORS = {
    "primary": "#00D2FF",       # Electric Cyan (Primary Brand / Selected State)
    "primary_alt": "#38BDF8",   # Sky Blue (Secondary Highlight / Curves)
    "strong": "#10B981",        # Emerald (Strong Match / Compliant Status)
    "possible": "#F59E0B",      # Amber (Possible Match / Attention Needed)
    "weak": "#EF4444",          # Coral Red (Weak Match / Disparity Detected)
    "bg": "#111827",            # Dark Navy Background
    "surface": "#1E293B",       # Slate Surface (Cards, Panels, Pills)
    "border": "#334155",        # Subtle Slate Divider & Border
    "text": "#F1F5F9",          # High-Contrast Light Slate Text
    "muted": "#94A3B8",         # Muted Secondary Gray Text
    "grid": "#334155",          # Chart Gridlines
}

# Decision-Support Match Tier Mappings (Colorblind Accessible)
TIER_COLORS = {
    "Strong Match": COLORS["strong"],
    "Possible Match": COLORS["possible"],
    "Weak Match": COLORS["weak"],
    "✓ Strong Match": COLORS["strong"],
    "⚠ Possible Match": COLORS["possible"],
    "✕ Weak Match": COLORS["weak"]
}

# Colorblind Accessible Status Indicators
COLORBLIND_LABELS = {
    "compliant": "✓ Compliant",
    "attention": "⚠ Attention",
    "disparity": "✕ Disparity",
    "strong": "✓ Strong Match",
    "possible": "⚠ Possible Match",
    "weak": "✕ Weak Match"
}

# -----------------------------------------------------------------------------
# 2. PLOTLY INTERACTIVE TEMPLATE & CONFIGURATION
# -----------------------------------------------------------------------------
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

def register_plotly_theme():
    """Registers and sets the candidate_dark Plotly template."""
    candidate_dark = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Plus Jakarta Sans, sans-serif",
                color=COLORS["text"],
                size=12
            ),
            title=dict(
                font=dict(
                    family="Plus Jakarta Sans, sans-serif",
                    color="#FFFFFF",
                    size=14,
                    weight="bold"
                ),
                x=0.02,
                xanchor="left"
            ),
            xaxis=dict(
                gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["border"],
                tickfont=dict(color=COLORS["muted"], family="Plus Jakarta Sans, sans-serif", size=11),
                title=dict(font=dict(color=COLORS["muted"], family="Plus Jakarta Sans, sans-serif", size=12))
            ),
            yaxis=dict(
                gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["border"],
                tickfont=dict(color=COLORS["text"], family="Plus Jakarta Sans, sans-serif", size=11),
                title=dict(font=dict(color=COLORS["muted"], family="Plus Jakarta Sans, sans-serif", size=12)),
                automargin=True
            ),
            legend=dict(
                font=dict(color=COLORS["text"], family="Plus Jakarta Sans, sans-serif", size=11),
                bgcolor="rgba(30, 41, 59, 0.8)",
                bordercolor=COLORS["border"],
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=50, b=40)
        )
    )
    pio.templates["candidate_dark"] = candidate_dark
    pio.templates.default = "candidate_dark"

register_plotly_theme()

# -----------------------------------------------------------------------------
# 3. GLOBAL CSS INJECTION
# -----------------------------------------------------------------------------
def inject_css():
    """Injects central high-contrast dark design system, typography, and tab styles."""
    st.markdown(f"""
    <style>
        /* Typography: Google Fonts Plus Jakarta Sans & Material Symbols */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        /* Scoped typography to prevent overriding icon ligature fonts */
        html, body, .stApp {{
            background-color: {COLORS['bg']} !important;
            color: {COLORS['text']} !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }}

        .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, input, textarea {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }}

        code, pre {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* PRESERVE MATERIAL SYMBOLS: Crucial for Streamlit sidebar collapse arrow */
        [data-testid="stIconMaterial"],
        [data-testid*="Icon"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stSidebarCollapseButton"] div,
        [data-testid="stSidebarNavCollapseButton"],
        [data-testid="stSidebarNavCollapseButton"] span,
        [data-testid="stSidebarHeader"] button span,
        .material-symbols-rounded,
        .material-symbols-outlined {{
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 1.25rem !important;
            display: inline-block !important;
            -webkit-font-feature-settings: 'liga' 1 !important;
            font-feature-settings: 'liga' 1 !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['surface']} !important;
            border-right: 1px solid {COLORS['border']} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
            color: {COLORS['text']} !important;
        }}

        /* SEGMENTED HORIZONTAL PILL TABS
           Note: Selectors [data-baseweb="tab"], [data-baseweb="tab-highlight"],
           and [data-baseweb="tab-border"] target BaseWeb UI components in Streamlit.
           These may need verification upon major Streamlit framework upgrades. */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {COLORS['surface']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 12px !important;
            padding: 6px !important;
            gap: 8px !important;
            margin-bottom: 1.3rem !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent !important;
            color: {COLORS['muted']} !important;
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            border: 1px solid transparent !important;
            transition: all 0.2s ease !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: #FFFFFF !important;
            background-color: rgba(51, 65, 85, 0.4) !important;
        }}

        /* Active Tab: Electric Cyan Background with Dark Navy Text (No default underline) */
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['primary']} !important;
            color: {COLORS['bg']} !important;
            font-weight: 700 !important;
            border: 1px solid {COLORS['primary']} !important;
            box-shadow: 0 0 14px rgba(0, 210, 255, 0.35) !important;
        }}

        /* Remove default BaseWeb underline indicator */
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
            height: 0px !important;
            background-color: transparent !important;
        }}

        /* Cards & Dark Panels */
        .dark-panel {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1.3rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        }}

        .panel-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Top KPI Metric Cards - Enforce Single Row (Both Sidebar Active & Inactive) */
        .kpi-container {{
            display: grid !important;
            grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
            gap: 0.65rem !important;
            margin-bottom: 1.2rem !important;
            width: 100% !important;
        }}

        .kpi-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.75rem 0.75rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.35);
            transition: transform 0.15s ease, border-color 0.15s ease;
            min-width: 0 !important;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .kpi-card:hover {{
            border-color: {COLORS['primary']};
            transform: translateY(-2px);
        }}

        .kpi-title {{
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS['muted']};
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.15;
            margin-bottom: 0.3rem;
            white-space: nowrap;
        }}

        .kpi-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.65rem;
            font-weight: 600;
            color: {COLORS['strong']};
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 2px 7px;
            border-radius: 9999px;
            white-space: nowrap;
            width: fit-content;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .kpi-badge.cyan {{
            color: {COLORS['primary_alt']};
            background: rgba(14, 165, 233, 0.14);
            border: 1px solid rgba(14, 165, 233, 0.3);
        }}

        .kpi-badge.purple {{
            color: #C084FC;
            background: rgba(168, 85, 247, 0.14);
            border: 1px solid rgba(168, 85, 247, 0.3);
        }}

        /* 6-Column KPI Grid for Model Performance (Preserves Score Visibility on Zoom) */
        .kpi-grid-6 {{
            display: grid !important;
            grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
            gap: 0.6rem !important;
            margin-bottom: 1.25rem !important;
            width: 100% !important;
        }}

        .kpi-card-compact {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.75rem 0.65rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.35);
            transition: transform 0.15s ease, border-color 0.15s ease;
            min-width: 0 !important;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .kpi-card-compact:hover {{
            border-color: {COLORS['primary']};
            transform: translateY(-2px);
        }}

        .kpi-card-compact .kpi-title {{
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS['muted']};
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .kpi-card-compact .kpi-value {{
            font-size: clamp(1.15rem, 1.45vw, 1.55rem);
            font-weight: 800;
            color: {COLORS['primary']};
            line-height: 1.15;
            margin-bottom: 0.3rem;
            white-space: nowrap;
        }}

        /* Mandatory Human Oversight Compliance Banner */
        .compliance-banner {{
            background: linear-gradient(90deg, rgba(0, 210, 255, 0.12), rgba(30, 41, 59, 0.8));
            border-left: 4px solid {COLORS['primary']};
            border-top: 1px solid rgba(0, 210, 255, 0.25);
            border-right: 1px solid rgba(0, 210, 255, 0.25);
            border-bottom: 1px solid rgba(0, 210, 255, 0.25);
            padding: 0.8rem 1.1rem;
            border-radius: 8px;
            color: {COLORS['text']};
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
            font-weight: 500;
        }}

        /* Colorblind-Safe Match Tier Badges (Always includes icon/label) */
        .tier-badge-strong {{
            background-color: rgba(16, 185, 129, 0.2);
            color: {COLORS['strong']};
            border: 1px solid rgba(16, 185, 129, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .tier-badge-possible {{
            background-color: rgba(245, 158, 11, 0.2);
            color: {COLORS['possible']};
            border: 1px solid rgba(245, 158, 11, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .tier-badge-weak {{
            background-color: rgba(239, 68, 68, 0.2);
            color: {COLORS['weak']};
            border: 1px solid rgba(239, 68, 68, 0.45);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }}

        /* Form Controls */
        .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox > div, .stMultiSelect > div {{
            background-color: {COLORS['surface']} !important;
            border: 1px solid {COLORS['border']} !important;
            color: {COLORS['text']} !important;
            border-radius: 8px !important;
        }}

        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 1px {COLORS['primary']} !important;
        }}

        /* Buttons */
        .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_alt']}) !important;
            color: {COLORS['bg']} !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 10px rgba(0, 210, 255, 0.35) !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{
            box-shadow: 0 4px 16px rgba(0, 210, 255, 0.5) !important;
            transform: translateY(-1px);
        }}

        .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {{
            background-color: {COLORS['surface']} !important;
            color: {COLORS['text']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 8px !important;
        }}

        /* Metric Widgets */
        [data-testid="stMetric"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS['primary']} !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS['muted']} !important;
            font-weight: 600 !important;
        }}

        /* Dataframe Containers */
        .stDataFrame {{
            border: 1px solid {COLORS['border']} !important;
            border-radius: 8px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

