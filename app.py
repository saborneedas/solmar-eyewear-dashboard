import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Solmar Eyewear | Growth Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 18px 20px;
        border-radius: 14px;
        background: #f7f8fa;
        border: 1px solid #e7e9ee;
        min-height: 110px;
    }
    .metric-label {font-size: 0.82rem; color: #667085; margin-bottom: 6px;}
    .metric-value {font-size: 1.75rem; font-weight: 700; color: #101828;}
    .insight {
        padding: 16px 18px;
        border-left: 4px solid #101828;
        background: #f8f9fb;
        border-radius: 8px;
        margin: 8px 0;
    }
    .small-note {color:#667085; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("dashboard_data.csv was not found. Place it in the same folder as app.py.")
    st.stop()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Solmar Intelligence")
st.sidebar.caption("Weekly growth & marketing analytics")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

view = st.sidebar.radio(
    "View",
    [
        "Executive Overview",
        "Marketing Intelligence",
        "Customer & Promotions",
        "Statistical Intelligence",
    ],
)

d = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if d.empty:
    st.warning("No observations exist for the selected period.")
    st.stop()

# -----------------------------
# HELPERS
# -----------------------------
def money(x):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"

def pct(x):
    return f"{x*100:.1f}%"

def metric(label, value):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def add_event_markers(fig, data):
    events = data[data["BFCM_promo"] == 1]
    for _, r in events.iterrows():
        fig.add_vline(
            x=r["date"],
            line_dash="dot",
            line_width=1,
            opacity=0.5
        )

# -----------------------------
# HEADER
# -----------------------------
st.title("SOLMAR EYEWEAR")
st.caption(
    f"Growth Intelligence Dashboard  •  "
    f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}"
)

# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================
if view == "Executive Overview":

    st.subheader("Executive Overview")

    total_rev = d["total_revenue"].sum()
    total_spend = d["total_media_spend"].sum()
    mer = total_rev / total_spend if total_spend else np.nan
    new_share = d["new_customer_revenue"].sum() / total_rev
    repeat_share = d["repeat_customer_revenue"].sum() / total_rev

    cols = st.columns(5)
    with cols[0]: metric("TOTAL REVENUE", money(total_rev))
    with cols[1]: metric("MEDIA SPEND", money(total_spend))
    with cols[2]: metric("OBSERVED MER", f"{mer:.2f}×")
    with cols[3]: metric("NEW CUSTOMER SHARE", pct(new_share))
    with cols[4]: metric("REPEAT CUSTOMER SHARE", pct(repeat_share))

    st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["total_revenue"],
        mode="lines", name="Total Revenue",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["new_customer_revenue"],
        mode="lines", name="New Customer",
        line=dict(width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["repeat_customer_revenue"],
        mode="lines", name="Repeat Customer",
        line=dict(width=1.5)
    ))
    add_event_markers(fig, d)
    fig.update_layout(
        title="Revenue Trajectory",
        yaxis_title="Revenue ($)",
        xaxis_title=None,
        hovermode="x unified",
        height=470,
        margin=dict(l=20,r=20,t=55,b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        channel_spend = pd.DataFrame({
            "Channel": ["Meta", "Google", "Other"],
            "Spend": [
                d["Meta_Spend"].sum(),
                d["Google_Spend"].sum(),
                d["Other_Spend"].sum()
            ]
        })
        fig = px.pie(
            channel_spend,
            names="Channel",
            values="Spend",
            hole=0.58,
            title="Media Spend Allocation"
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            d,
            x="BFCM_promo",
            y="total_revenue",
            points="outliers",
            title="Revenue Distribution: BFCM vs Non-BFCM"
        )
        fig.update_xaxes(
            tickvals=[0,1],
            ticktext=["Non-BFCM", "BFCM"]
        )
        fig.update_layout(height=390, yaxis_title="Weekly Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What the data says")

    promo_med = d.loc[d["Promos"] == 0, "total_revenue"].median()
    promo_active = d.loc[d["Promos"] > 0, "total_revenue"].median()

    insights = [
        f"Repeat customers contribute {pct(repeat_share)} of revenue over the selected period.",
        f"Media spend is concentrated in Meta ({d['Meta_Spend'].sum()/total_spend:.1%}) and Google ({d['Google_Spend'].sum()/total_spend:.1%}).",
        f"Observed revenue-to-media-spend ratio (MER) is {mer:.2f}× for the selected period.",
    ]

    if not pd.isna(promo_med) and not pd.isna(promo_active):
        insights.append(
            f"Median revenue is {money(promo_active)} in weeks with promotions versus "
            f"{money(promo_med)} in weeks with no promotions."
        )

    for x in insights:
        st.markdown(f'<div class="insight">{x}</div>', unsafe_allow_html=True)

    st.caption(
        "Analytical caution: observational associations are not equivalent to causal attribution."
    )

# ============================================================
# MARKETING INTELLIGENCE
# ============================================================
elif view == "Marketing Intelligence":

    st.subheader("Marketing Intelligence")

    total_spend = d["total_media_spend"].sum()
    mer = d["total_revenue"].sum() / total_spend

    cols = st.columns(4)
    with cols[0]: metric("MEDIA SPEND", money(total_spend))
    with cols[1]: metric("OBSERVED MER", f"{mer:.2f}×")
    with cols[2]: metric("MEDIAN WEEKLY MER", f"{d['MER'].median():.2f}×")
    with cols[3]: metric("PEAK WEEKLY MER", f"{d['MER'].max():.2f}×")

    st.divider()

    channel = st.selectbox("Channel", ["All Channels", "Meta", "Google", "Other"])

    if channel == "All Channels":
        spend_cols = ["Meta_Spend", "Google_Spend", "Other_Spend"]
        labels = ["Meta", "Google", "Other"]
        temp = pd.DataFrame({
            "date": d["date"],
            "Meta": d["Meta_Spend"],
            "Google": d["Google_Spend"],
            "Other": d["Other_Spend"],
        }).melt("date", var_name="Channel", value_name="Spend")

        fig = px.line(
            temp, x="date", y="Spend", color="Channel",
            title="Weekly Media Spend by Channel"
        )
        fig.update_layout(height=430, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    else:
        col = f"{channel}_Spend"
        fig = px.line(
            d, x="date", y=col,
            title=f"Weekly {channel} Spend"
        )
        fig.update_layout(height=430, yaxis_title="Spend ($)")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        corr = d[["total_revenue","Meta_Spend","Google_Spend","Other_Spend"]].corr()["total_revenue"].drop("total_revenue")
        corr_df = corr.reset_index()
        corr_df.columns = ["Channel", "Correlation"]
        fig = px.bar(
            corr_df,
            x="Channel",
            y="Correlation",
            text_auto=".3f",
            title="Contemporaneous Revenue Association"
        )
        fig.update_yaxes(range=[0,1])
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        lag_data = pd.DataFrame({
            "Lag": [0,1,2,3,4],
            "Meta": [0.914,0.616,0.230,0.175,0.086],
            "Google": [0.820,0.501,0.290,0.252,0.175],
            "Other": [0.724,0.552,0.312,0.272,0.154],
        }).melt("Lag", var_name="Channel", value_name="Correlation")

        fig = px.line(
            lag_data,
            x="Lag",
            y="Correlation",
            color="Channel",
            markers=True,
            title="Revenue Association Across Lags"
        )
        fig.update_layout(height=390, xaxis_title="Lag (weeks)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Marketing Efficiency")

    fig = px.line(
        d, x="date", y="MER",
        title="Observed Revenue / Media Spend"
    )
    fig.add_hline(
        y=d["MER"].median(),
        line_dash="dash",
        annotation_text=f"Median: {d['MER'].median():.2f}×"
    )
    fig.update_layout(height=420, yaxis_title="MER (×)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "MER is a business-level efficiency ratio. It should not be interpreted as "
        "channel-level ROAS because the dataset does not contain channel-attributed revenue."
    )

# ============================================================
# CUSTOMER & PROMOTIONS
# ============================================================
elif view == "Customer & Promotions":

    st.subheader("Customer & Promotion Intelligence")

    total = d["total_revenue"].sum()
    new_share = d["new_customer_revenue"].sum() / total
    repeat_share = d["repeat_customer_revenue"].sum() / total

    cols = st.columns(4)
    with cols[0]: metric("NEW REVENUE", money(d["new_customer_revenue"].sum()))
    with cols[1]: metric("REPEAT REVENUE", money(d["repeat_customer_revenue"].sum()))
    with cols[2]: metric("NEW SHARE", pct(new_share))
    with cols[3]: metric("REPEAT SHARE", pct(repeat_share))

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["new_revenue_share"]*100,
            mode="lines", name="New"
        ))
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["repeat_revenue_share"]*100,
            mode="lines", name="Repeat"
        ))
        fig.add_hline(y=50, line_dash="dash")
        fig.update_layout(
            title="Customer Revenue Mix",
            yaxis_title="Revenue Share (%)",
            height=410,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        promo = d.copy()
        promo["Promo Group"] = pd.cut(
            promo["Promos"],
            bins=[-1,0,2,np.inf],
            labels=["No Promo","1–2 Promos","3+ Promos"]
        )
        fig = px.box(
            promo,
            x="Promo Group",
            y="total_revenue",
            points="outliers",
            title="Revenue Distribution by Promotion Intensity"
        )
        fig.update_layout(height=410, yaxis_title="Weekly Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        event = d.groupby("BFCM_promo")["total_revenue"].median().reset_index()
        event["Label"] = event["BFCM_promo"].map({0:"Non-BFCM",1:"BFCM"})
        fig = px.bar(
            event, x="Label", y="total_revenue",
            text_auto=".3s",
            title="Median Revenue: BFCM vs Non-BFCM"
        )
        fig.update_layout(height=360, yaxis_title="Median Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        event = d.groupby("Price_Hike")["total_revenue"].median().reset_index()
        event["Label"] = event["Price_Hike"].map({0:"Pre Price Hike",1:"Post Price Hike"})
        fig = px.bar(
            event, x="Label", y="total_revenue",
            text_auto=".3s",
            title="Median Revenue: Price-Hike Period"
        )
        fig.update_layout(height=360, yaxis_title="Median Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer movement")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["date"],
        y=d["new_customer_revenue"],
        name="New Customer"
    ))
    fig.add_trace(go.Bar(
        x=d["date"],
        y=d["repeat_customer_revenue"],
        name="Repeat Customer"
    ))
    fig.update_layout(
        barmode="stack",
        title="Weekly Revenue Composition",
        height=430,
        yaxis_title="Revenue ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "BFCM has only four observations in the full dataset; event-level comparisons "
        "should therefore be interpreted cautiously."
    )

# ============================================================
# STATISTICAL INTELLIGENCE
# ============================================================
else:

    st.subheader("Statistical Intelligence")

    st.markdown(
        "This section exposes the statistical evidence behind the dashboard rather "
        "than hiding it behind visualisations."
    )

    # Correlations
    st.markdown("### Revenue associations")

    corr = d[
        [
            "total_revenue",
            "Meta_Spend",
            "Google_Spend",
            "Other_Spend",
            "total_media_spend",
            "Promos"
        ]
    ].corr()["total_revenue"].sort_values(ascending=False)

    corr_df = corr.reset_index()
    corr_df.columns = ["Variable", "Pearson Correlation"]
    corr_df = corr_df[corr_df["Variable"] != "total_revenue"]

    st.dataframe(
        corr_df.style.format({"Pearson Correlation": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Regression evidence")

    regression = pd.DataFrame({
        "Variable": [
            "Meta Spend",
            "Google Spend",
            "Other Spend",
            "Promos",
            "BFCM Promo",
            "Price Hike",
            "Time Index"
        ],
        "Coefficient": [
            4.7866,
            5.0913,
            -1.3088,
            30043.4759,
            311567.6031,
            -131397.9807,
            100.3629
        ],
        "p-value": [
            0.0000,
            0.0000,
            0.5465,
            0.1668,
            0.0370,
            0.1356,
            0.9431
        ],
        "CI Lower": [
            3.6309,
            2.9143,
            -5.6051,
            -12781.9469,
            19245.2442,
            -304766.5667,
            -2687.8465
        ],
        "CI Upper": [
            5.9424,
            7.2682,
            2.9874,
            72868.8987,
            603889.9620,
            41970.6053,
            2888.5723
        ]
    })

    st.dataframe(
        regression.style.format({
            "Coefficient": "{:,.2f}",
            "p-value": "{:.4f}",
            "CI Lower": "{:,.2f}",
            "CI Upper": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("R²", "0.919")
    with c2:
        metric("ADJUSTED R²", "0.903")
    with c3:
        metric("OBSERVATIONS", "109")

    st.markdown("### Model diagnostics")

    diagnostics = pd.DataFrame({
        "Diagnostic": [
            "Breusch–Godfrey",
            "Breusch–Pagan",
            "Highest VIF",
            "Durbin–Watson"
        ],
        "Result": [
            "p = 0.0396",
            "p = 0.0267",
            "7.15 (time index)",
            "1.712"
        ],
        "Interpretation": [
            "Evidence of serial correlation",
            "Evidence of heteroskedasticity",
            "Potential multicollinearity",
            "Residual dependence warrants attention"
        ]
    })

    st.dataframe(diagnostics, use_container_width=True, hide_index=True)

    st.warning(
        "The OLS model has diagnostic limitations: evidence of autocorrelation, "
        "heteroskedasticity and multicollinearity. Coefficients should therefore "
        "be interpreted as conditional associations, not causal effects."
    )

    st.markdown("### Revenue anomalies")

    anomaly_cols = [
        "date", "total_revenue", "total_media_spend",
        "MER", "Promos", "BFCM_promo", "Price_Hike",
        "revenue_robust_z"
    ]

    anomalies = d[d["anomaly"] == "Anomaly"][anomaly_cols].copy()
    anomalies = anomalies.sort_values(
        "revenue_robust_z",
        key=lambda x: x.abs(),
        ascending=False
    )

    if anomalies.empty:
        st.success("No robust revenue anomalies detected in the selected period.")
    else:
        st.dataframe(
            anomalies.style.format({
                "total_revenue": "${:,.0f}",
                "total_media_spend": "${:,.0f}",
                "MER": "{:.2f}×",
                "revenue_robust_z": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Anomaly detection uses a rolling median/MAD framework and is intended for "
        "investigation, not automatic causal explanation."
    )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption(
    "Solmar Eyewear — Growth Intelligence | Weekly observational dataset | "
    "Statistics-first decision support"
)
