import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SOLMAR | Growth Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================
ACCENT = "#8B5CF6"
ACCENT2 = "#22D3EE"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#FB7185"
BG = "#070B14"
PANEL = "#0D1321"
CARD = "#111827"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BORDER = "#243044"
GRID = "#1E293B"

st.markdown(
    f"""
<style>
:root {{ color-scheme: dark; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding: 1.1rem 2.2rem 2.5rem; max-width: 1700px; }}
[data-testid="stSidebar"] {{ background: #090E19; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}

.hero {{
    background: linear-gradient(135deg, #111827 0%, #0B1220 55%, #15102A 100%);
    border: 1px solid {BORDER}; border-radius: 22px; padding: 25px 28px;
    margin-bottom: 18px; position: relative; overflow: hidden;
}}
.hero:after {{ content:""; position:absolute; width:280px; height:280px; right:-100px; top:-140px;
    border-radius:50%; background: radial-gradient(circle, rgba(139,92,246,.22), transparent 68%); }}
.hero-kicker {{ color: {ACCENT2}; font-size:.75rem; letter-spacing:.18em; font-weight:800; }}
.hero-title {{ color:{TEXT}; font-size:2.25rem; font-weight:850; margin:.2rem 0; letter-spacing:-.04em; }}
.hero-sub {{ color:{MUTED}; font-size:.92rem; }}

.section {{ color:{TEXT}; font-size:1.18rem; font-weight:800; margin: 1.0rem 0 .55rem; }}
.section-note {{ color:{MUTED}; font-size:.82rem; margin-bottom:.8rem; }}

.kpi {{ background: linear-gradient(145deg, {CARD}, #0C1320); border:1px solid {BORDER};
    border-radius:16px; padding:16px 17px; min-height:108px; }}
.kpi-label {{ color:{MUTED}; font-size:.70rem; letter-spacing:.11em; font-weight:800; }}
.kpi-value {{ color:{TEXT}; font-size:1.65rem; font-weight:850; margin-top:7px; }}
.kpi-delta {{ color:{GREEN}; font-size:.76rem; margin-top:4px; }}

.insight {{ background:{PANEL}; border:1px solid {BORDER}; border-left:3px solid {ACCENT};
    border-radius:12px; padding:13px 15px; margin:7px 0; color:{TEXT}; }}
.insight b {{ color:{ACCENT2}; }}
.callout {{ background:rgba(139,92,246,.09); border:1px solid rgba(139,92,246,.28); border-radius:14px;
    padding:15px 17px; color:#EDE9FE; }}
.caution {{ background:rgba(251,191,36,.07); border:1px solid rgba(251,191,36,.25); border-radius:14px;
    padding:13px 15px; color:#FEF3C7; }}
.good {{ background:rgba(52,211,153,.07); border:1px solid rgba(52,211,153,.22); border-radius:14px;
    padding:13px 15px; color:#D1FAE5; }}

div[data-testid="stTabs"] button {{ color:{MUTED}; font-weight:700; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color:{TEXT}; }}
button[kind="secondary"] {{ border-color:{BORDER}; }}

.small {{ color:{MUTED}; font-size:.78rem; }}
.big-number {{ font-size:2rem; font-weight:850; color:{TEXT}; }}

[data-testid="stMetricValue"] {{ color:{TEXT}; }}
[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("dashboard_data.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Rebuild any derived fields if the exported CSV does not contain them.
    meta_cols = [c for c in df.columns if c.startswith("spend_Meta_")]
    google_cols = [c for c in df.columns if c.startswith("spend_google_")]
    other_cols = [c for c in ["spend_snapchat", "spend_Youtube", "Tiktok_DTC_Spends"] if c in df.columns]

    if "Meta_Spend" not in df.columns and meta_cols:
        df["Meta_Spend"] = df[meta_cols].sum(axis=1)
    if "Google_Spend" not in df.columns and google_cols:
        df["Google_Spend"] = df[google_cols].sum(axis=1)
    if "Other_Spend" not in df.columns and other_cols:
        df["Other_Spend"] = df[other_cols].sum(axis=1)
    if "total_media_spend" not in df.columns:
        df["total_media_spend"] = df[["Meta_Spend", "Google_Spend", "Other_Spend"]].sum(axis=1)
    if "MER" not in df.columns:
        df["MER"] = df["total_revenue"] / df["total_media_spend"].replace(0, np.nan)
    if "new_revenue_share" not in df.columns:
        df["new_revenue_share"] = df["new_customer_revenue"] / df["total_revenue"].replace(0, np.nan)
    if "repeat_revenue_share" not in df.columns:
        df["repeat_revenue_share"] = df["repeat_customer_revenue"] / df["total_revenue"].replace(0, np.nan)
    if "revenue_change" not in df.columns:
        df["revenue_change"] = df["total_revenue"].diff()

    # Robust rolling anomaly score; no distributional normality assumption.
    med = df["total_revenue"].rolling(13, center=True, min_periods=7).median()
    mad = (df["total_revenue"] - med).abs().rolling(13, center=True, min_periods=7).median()
    df["revenue_robust_z"] = (df["total_revenue"] - med) / (1.4826 * mad.replace(0, np.nan))
    df["anomaly"] = np.where(df["revenue_robust_z"].abs() >= 3, "Anomaly", "Normal")

    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["year"] = df["date"].dt.year
    df["period_label"] = df["date"].dt.strftime("%d %b %Y")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dashboard_data.csv: {e}")
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def money(x):
    if pd.isna(x): return "—"
    x = float(x)
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000: return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"

def money_full(x):
    return "—" if pd.isna(x) else f"${float(x):,.0f}"

def pct(x):
    return "—" if pd.isna(x) else f"{float(x)*100:.1f}%"

def num(x):
    return "—" if pd.isna(x) else f"{float(x):,.2f}"

def kpi(label, value, delta=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )

def plot_layout(fig, height=430):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=58, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111827", font_color=TEXT, bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

def add_events(fig, data):
    for _, r in data[data["BFCM_promo"] == 1].iterrows():
        fig.add_vline(x=r["date"], line_dash="dot", line_width=1, line_color=AMBER, opacity=.55)

def selection_index(event):
    try:
        points = event.selection.points
        if not points: return None
        p = points[0]
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple, np.ndarray)):
            cd = cd[0]
        if cd is not None:
            return int(cd)
        return int(p.get("point_index", p.get("point_number")))
    except Exception:
        return None

@st.dialog("Data Point Intelligence", width="large")
def point_dialog(row, context="Selected observation"):
    st.caption(context)
    st.markdown(f"### {row['date'].strftime('%A, %d %B %Y')}")
    cols = st.columns(4)
    with cols[0]: st.metric("Revenue", money(row["total_revenue"]))
    with cols[1]: st.metric("Media spend", money(row["total_media_spend"]))
    with cols[2]: st.metric("Observed MER", f"{row['MER']:.2f}×")
    with cols[3]: st.metric("Promotions", int(row["Promos"]))

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("**Customer economics**")
        st.write(f"New-customer revenue: **{money(row['new_customer_revenue'])}**")
        st.write(f"Repeat-customer revenue: **{money(row['repeat_customer_revenue'])}**")
        st.write(f"New-customer share: **{pct(row['new_revenue_share'])}**")
        st.write(f"Repeat-customer share: **{pct(row['repeat_revenue_share'])}**")
    with b:
        st.markdown("**Marketing & events**")
        st.write(f"Meta: **{money(row['Meta_Spend'])}**")
        st.write(f"Google: **{money(row['Google_Spend'])}**")
        st.write(f"Other: **{money(row['Other_Spend'])}**")
        st.write(f"BFCM: **{'Yes' if row['BFCM_promo'] else 'No'}**  ·  Price hike: **{'Yes' if row['Price_Hike'] else 'No'}**")

    z = row.get("revenue_robust_z", np.nan)
    if pd.notna(z) and abs(z) >= 3:
        st.markdown(f'<div class="caution"><b>Statistical flag:</b> this week is a robust revenue anomaly (score {z:.1f}). This is a signal to investigate, not a causal explanation.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good"><b>Reading this point:</b> compare the revenue level with its spend, customer mix, promotion count and event flags before drawing conclusions.</div>', unsafe_allow_html=True)


def interactive_chart(fig, data, key, context="Selected observation", height=430):
    fig = plot_layout(fig, height)
    fig.update_traces(customdata=np.arange(len(data)).reshape(-1, 1))
    event = st.plotly_chart(
        fig, use_container_width=True, key=key,
        on_select="rerun", selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToAdd": ["drawline", "eraseshape"]},
    )
    idx = selection_index(event)
    if idx is not None and 0 <= idx < len(data):
        point_dialog(data.iloc[idx], context)
    return event

# ============================================================
# SIDEBAR — GLOBAL CROSS FILTERS
# ============================================================
st.sidebar.markdown("## ◈ SOLMAR INTELLIGENCE")
st.sidebar.caption("Interactive weekly growth analytics")
st.sidebar.divider()

min_date, max_date = df["date"].min().date(), df["date"].max().date()
dates = st.sidebar.date_input("Analysis window", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(dates, tuple) and len(dates) == 2:
    start_date, end_date = dates
else:
    start_date, end_date = min_date, max_date

event_filter = st.sidebar.multiselect("Event filters", ["BFCM", "Price hike", "Any promotion"], default=[])
promo_group = st.sidebar.multiselect("Promotion intensity", ["No promo", "1–2 promos", "3+ promos"], default=[])

pages = ["Command Center", "Marketing Lab", "Customer & Events", "Statistical Intelligence", "Anomaly Radar"]
page = st.sidebar.radio("Navigate", pages, index=0)

# Apply filters
d = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()
if "BFCM" in event_filter: d = d[d["BFCM_promo"] == 1]
if "Price hike" in event_filter: d = d[d["Price_Hike"] == 1]
if "Any promotion" in event_filter: d = d[d["Promos"] > 0]
if promo_group:
    masks = []
    for g in promo_group:
        if g == "No promo": masks.append(d["Promos"] == 0)
        elif g == "1–2 promos": masks.append(d["Promos"].between(1, 2))
        else: masks.append(d["Promos"] >= 3)
    d = d[np.logical_or.reduce(masks)]

if d.empty:
    st.warning("No observations match the current filters. Broaden the filters to continue.")
    st.stop()

# ============================================================
# HERO
# ============================================================
st.markdown(
    f'<div class="hero"><div class="hero-kicker">SOLMAR EYEWEAR · GROWTH INTELLIGENCE</div>'
    f'<div class="hero-title">The business, one interaction at a time.</div>'
    f'<div class="hero-sub">{start_date.strftime("%d %b %Y")} → {end_date.strftime("%d %b %Y")} · '
    f'{len(d)} weekly observations · Click any chart point to open its full intelligence card.</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# COMMAND CENTER
# ============================================================
if page == "Command Center":
    total = d["total_revenue"].sum(); spend = d["total_media_spend"].sum()
    mer = total / spend if spend else np.nan
    new_share = d["new_customer_revenue"].sum() / total
    repeat_share = d["repeat_customer_revenue"].sum() / total
    peak = d.loc[d["total_revenue"].idxmax()]

    cols = st.columns(5)
    with cols[0]: kpi("TOTAL REVENUE", money(total), f"Peak week {peak['date'].strftime('%d %b %y')}")
    with cols[1]: kpi("MEDIA SPEND", money(spend), f"{spend/len(d):,.0f} avg / week")
    with cols[2]: kpi("OBSERVED MER", f"{mer:.2f}×", "Revenue ÷ media spend")
    with cols[3]: kpi("NEW CUSTOMER MIX", pct(new_share), "Period revenue share")
    with cols[4]: kpi("REPEAT CUSTOMER MIX", pct(repeat_share), "Period revenue share")

    st.markdown('<div class="section">Revenue cockpit</div><div class="section-note">Hover for context. <b>Click a point</b> for a full weekly intelligence pop-up. Drag to zoom; double-click to reset.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for col, name, width in [("total_revenue", "Total revenue", 4), ("new_customer_revenue", "New customer", 2), ("repeat_customer_revenue", "Repeat customer", 2)]:
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], mode="lines+markers", name=name, line=dict(width=width), marker=dict(size=7), customdata=np.arange(len(d)).reshape(-1,1), hovertemplate="<b>%{x|%d %b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>"))
    add_events(fig, d)
    interactive_chart(fig, d, "cmd_revenue", "Revenue trajectory — weekly drill-down", 510)

    a,b,c = st.columns([1.05,1,1])
    with a:
        ch = pd.DataFrame({"Channel":["Meta","Google","Other"],"Spend":[d.Meta_Spend.sum(),d.Google_Spend.sum(),d.Other_Spend.sum()]})
        fig = px.pie(ch, names="Channel", values="Spend", hole=.65, title="Media allocation")
        fig.update_traces(textinfo="label+percent", customdata=np.arange(len(ch)).reshape(-1,1))
        interactive_chart(fig, ch, "cmd_spend", "Channel allocation", 390)
    with b:
        temp=d[["date","MER"]].copy()
        fig=px.line(temp,x="date",y="MER",markers=True,title="Observed MER")
        fig.add_hline(y=temp.MER.median(),line_dash="dash",line_color=AMBER,annotation_text=f"Median {temp.MER.median():.2f}×")
        interactive_chart(fig,temp,"cmd_mer","MER weekly drill-down",390)
    with c:
        fig=px.histogram(d,x="total_revenue",nbins=16,title="Revenue distribution")
        fig.update_xaxes(title="Weekly revenue")
        fig.update_yaxes(title="Weeks")
        fig=plot_layout(fig,390)
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Executive interpretation</div>', unsafe_allow_html=True)
    promo0=d.loc[d.Promos==0,"total_revenue"].median(); promop=d.loc[d.Promos>0,"total_revenue"].median()
    st.markdown(f'<div class="insight"><b>Customer engine:</b> repeat customers represent <b>{pct(repeat_share)}</b> of period revenue, while new customers represent <b>{pct(new_share)}</b>.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight"><b>Marketing concentration:</b> Meta accounts for <b>{d.Meta_Spend.sum()/spend:.1%}</b> of observed media spend; Google accounts for <b>{d.Google_Spend.sum()/spend:.1%}</b>.</div>', unsafe_allow_html=True)
    if pd.notna(promo0) and pd.notna(promop):
        st.markdown(f'<div class="insight"><b>Promotion signal:</b> median weekly revenue is {money(promop)} in promotion-active weeks versus {money(promo0)} in no-promotion weeks. This is an association, not a causal estimate.</div>', unsafe_allow_html=True)

# ============================================================
# MARKETING LAB
# ============================================================
elif page == "Marketing Lab":
    st.markdown('<div class="section">Marketing Lab</div><div class="section-note">Explore spend, revenue association, lag structure and business-level efficiency. No channel ROAS is fabricated because channel-attributed revenue is not present.</div>', unsafe_allow_html=True)

    channel = st.selectbox("Focus channel", ["All channels", "Meta", "Google", "Other"], key="mkt_channel")
    lag = st.slider("Association lag (weeks)", 0, 4, 0, key="mkt_lag")

    spend_map={"Meta":"Meta_Spend","Google":"Google_Spend","Other":"Other_Spend"}
    if channel == "All channels":
        temp=d[["date","Meta_Spend","Google_Spend","Other_Spend"]].melt("date",var_name="Channel",value_name="Spend")
        temp["Channel"]=temp["Channel"].str.replace("_Spend","")
        fig=px.line(temp,x="date",y="Spend",color="Channel",markers=True,title="Weekly media spend — click a point")
        # use base d mapping for selection only; modal remains tied to week index
        interactive_chart(fig,d,"mkt_spend","Weekly marketing spend",450)
    else:
        col=spend_map[channel]
        fig=px.line(d,x="date",y=col,markers=True,title=f"Weekly {channel} spend — click a point")
        interactive_chart(fig,d,"mkt_spend_single",f"{channel} spend — weekly drill-down",450)

    a,b=st.columns(2)
    with a:
        corr_rows=[]
        for ch,col in spend_map.items():
            x=d[col]
            y=d.total_revenue
            if lag>0:
                x=x.shift(lag)
            mask=x.notna() & y.notna()
            r,p=stats.pearsonr(x[mask],y[mask]) if mask.sum()>=3 else (np.nan,np.nan)
            corr_rows.append({"Channel":ch,"Correlation":r,"p_value":p})
        corr_df=pd.DataFrame(corr_rows)
        fig=px.bar(corr_df,x="Channel",y="Correlation",text="Correlation",title=f"Revenue association at lag {lag}")
        fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        fig.update_yaxes(range=[-1,1],title="Pearson r")
        interactive_chart(fig,corr_df,"mkt_corr","Channel association",390)
    with b:
        lag_rows=[]
        for ch,col in spend_map.items():
            for l in range(5):
                x=d[col].shift(l); y=d.total_revenue
                mask=x.notna() & y.notna()
                r=stats.pearsonr(x[mask],y[mask])[0] if mask.sum()>=3 else np.nan
                lag_rows.append({"Channel":ch,"Lag":l,"Correlation":r})
        ld=pd.DataFrame(lag_rows)
        fig=px.line(ld,x="Lag",y="Correlation",color="Channel",markers=True,title="How the association changes with lag")
        fig.update_xaxes(dtick=1,title="Spend leads revenue by (weeks)")
        fig.update_yaxes(range=[-1,1],title="Pearson r")
        interactive_chart(fig,ld,"mkt_lag","Lag-response association",390)

    st.markdown('<div class="section">Spend vs revenue explorer</div>', unsafe_allow_html=True)
    xchannel=st.selectbox("X-axis spend",["Meta","Google","Other","Total media"],key="scatter_channel")
    xcol={"Meta":"Meta_Spend","Google":"Google_Spend","Other":"Other_Spend","Total media":"total_media_spend"}[xchannel]
    scat=d[["date",xcol,"total_revenue","MER","Promos","BFCM_promo"]].copy()
    fig=px.scatter(scat,x=xcol,y="total_revenue",size="MER",color="Promos",hover_data=["date","MER","BFCM_promo"],title=f"{xchannel} spend vs weekly revenue — click a point")
    interactive_chart(fig,d,"mkt_scatter","Spend vs revenue — observation drill-down",500)
    st.markdown('<div class="caution"><b>Statistical reading:</b> correlation measures co-movement, not incremental revenue caused by a channel. Spend can be increased in response to expected demand, creating reverse causality.</div>', unsafe_allow_html=True)

# ============================================================
# CUSTOMER & EVENTS
# ============================================================
elif page == "Customer & Events":
    st.markdown('<div class="section">Customer & Event Intelligence</div><div class="section-note">Interact with customer mix, promotion intensity, BFCM and price-hike periods. Small event samples are shown explicitly.</div>', unsafe_allow_html=True)

    mix=d[["date","new_customer_revenue","repeat_customer_revenue","new_revenue_share","repeat_revenue_share"]].copy()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=mix.date,y=mix.new_revenue_share*100,mode="lines+markers",name="New %",customdata=np.arange(len(mix)).reshape(-1,1)))
    fig.add_trace(go.Scatter(x=mix.date,y=mix.repeat_revenue_share*100,mode="lines+markers",name="Repeat %",customdata=np.arange(len(mix)).reshape(-1,1)))
    fig.add_hline(y=50,line_dash="dot",line_color=AMBER)
    interactive_chart(fig,d,"cust_mix","Customer mix — weekly drill-down",460)

    a,b=st.columns(2)
    with a:
        p=d.copy(); p["Promo group"]=pd.cut(p.Promos,bins=[-1,0,2,np.inf],labels=["No promo","1–2 promos","3+ promos"])
        fig=px.box(p,x="Promo group",y="total_revenue",points="all",color="Promo group",title="Revenue by promotion intensity")
        fig=plot_layout(fig,430)
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    with b:
        ev=d.copy(); ev["BFCM label"]=ev.BFCM_promo.map({0:"Non-BFCM",1:"BFCM"})
        fig=px.box(ev,x="BFCM label",y="total_revenue",points="all",color="BFCM label",title="BFCM revenue distribution")
        fig=plot_layout(fig,430)
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Event comparison, with statistical context</div>', unsafe_allow_html=True)
    event_choice=st.radio("Compare",["BFCM vs non-BFCM","Price hike vs pre-hike","Promotion active vs inactive"],horizontal=True,key="event_choice")
    if event_choice=="BFCM vs non-BFCM":
        flag=d.BFCM_promo; labels={0:"Non-BFCM",1:"BFCM"}
    elif event_choice=="Price hike vs pre-hike":
        flag=d.Price_Hike; labels={0:"Pre price hike",1:"Post price hike"}
    else:
        flag=(d.Promos>0).astype(int); labels={0:"No promotion",1:"Promotion active"}
    groups=[]
    for g in [0,1]:
        vals=d.loc[flag==g,"total_revenue"].dropna()
        groups.append({"Group":labels[g],"n":len(vals),"Median":vals.median() if len(vals) else np.nan,"Mean":vals.mean() if len(vals) else np.nan})
    comp=pd.DataFrame(groups)
    fig=px.bar(comp,x="Group",y="Median",text="n",title="Median weekly revenue — sample size shown on bars")
    fig.update_traces(texttemplate="n=%{text}",textposition="outside")
    fig=plot_layout(fig,380)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    vals0=d.loc[flag==0,"total_revenue"].dropna(); vals1=d.loc[flag==1,"total_revenue"].dropna()
    if len(vals0)>=3 and len(vals1)>=3:
        u,p=stats.mannwhitneyu(vals0,vals1,alternative="two-sided")
        st.markdown(f'<div class="callout"><b>Non-parametric test:</b> Mann–Whitney U = {u:.0f}, p = {p:.4f}. This tests whether the two groups differ in distribution; it does not prove that the event caused the difference.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="caution"><b>Not enough observations:</b> this comparison has fewer than 3 observations in at least one group after filtering.</div>',unsafe_allow_html=True)

# ============================================================
# STATISTICAL INTELLIGENCE
# ============================================================
elif page == "Statistical Intelligence":
    st.markdown('<div class="section">Statistical Intelligence — translated for everyone</div><div class="section-note">You do not need a statistics background. Start with <b>Plain English</b>; switch to <b>Analyst mode</b> when you want the formal statistic.</div>', unsafe_allow_html=True)
    mode=st.segmented_control("Interpretation mode",["Plain English","Analyst mode"],default="Plain English",key="stat_mode")

    var_options={
        "Meta spend":"Meta_Spend","Google spend":"Google_Spend","Other spend":"Other_Spend",
        "Total media spend":"total_media_spend","Promotions count":"Promos","MER":"MER",
        "New-customer share":"new_revenue_share"
    }
    variable=st.selectbox("Choose a variable to investigate against revenue",list(var_options),key="stat_var")
    xcol=var_options[variable]
    x=d[xcol]; y=d.total_revenue
    mask=x.notna() & y.notna()
    r,p=stats.pearsonr(x[mask],y[mask]) if mask.sum()>=3 else (np.nan,np.nan)
    rho,sp=stats.spearmanr(x[mask],y[mask]) if mask.sum()>=3 else (np.nan,np.nan)
    n=int(mask.sum())

    a,b,c,dcol=st.columns(4)
    with a:kpi("PEARSON r",f"{r:.3f}" if pd.notna(r) else "—","Linear association")
    with b:kpi("SPEARMAN ρ",f"{rho:.3f}" if pd.notna(rho) else "—","Rank association")
    with c:kpi("P-VALUE",f"{p:.4f}" if pd.notna(p) else "—","Test of zero Pearson correlation")
    with dcol:kpi("N",str(n),"Usable observations")

    scat=d[["date",xcol,"total_revenue","MER","Promos","BFCM_promo","Price_Hike"]].copy()
    fig=px.scatter(scat,x=xcol,y="total_revenue",size="MER",color="Promos",hover_data=["date","MER","BFCM_promo","Price_Hike"],title=f"{variable} vs revenue — click any observation")
    interactive_chart(fig,d,"stat_scatter","Statistical observation drill-down",500)

    if mode=="Plain English":
        strength="very weak" if abs(r)<.2 else "weak" if abs(r)<.4 else "moderate" if abs(r)<.7 else "strong" if abs(r)<.9 else "very strong"
        direction="moves upward" if r>0 else "moves downward" if r<0 else "has little linear movement"
        evidence="The observed relationship is statistically distinguishable from zero at the 5% level." if p<.05 else "The sample does not provide strong evidence of a non-zero linear relationship at the 5% level."
        direction_word = "positive" if r > 0 else "negative" if r < 0 else "near-zero"
        st.markdown(f'<div class="callout"><b>In plain English:</b> In these weeks, <b>{variable}</b> has a <b>{strength}</b> {direction_word} association with revenue and generally <b>{direction}</b> as revenue changes. {evidence}</div>',unsafe_allow_html=True)
        st.markdown('<div class="caution"><b>Important:</b> “associated with” is deliberately not written as “caused.” A high correlation can arise from seasonality, shared demand, reverse causality or omitted variables.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="callout"><b>Formal reading:</b> Pearson r = {r:.4f} tests linear association. Spearman ρ = {rho:.4f} tests monotonic rank association. Pearson p = {p:.4g} is the two-sided test of H₀: ρ = 0 under the usual correlation assumptions.</div>',unsafe_allow_html=True)

    st.markdown('<div class="section">Correlation matrix — interactive hover</div>',unsafe_allow_html=True)
    corr_cols=["total_revenue","Meta_Spend","Google_Spend","Other_Spend","total_media_spend","Promos","MER","new_revenue_share"]
    cm=d[corr_cols].corr()
    fig=go.Figure(go.Heatmap(z=cm.values,x=["Revenue","Meta","Google","Other","Total media","Promos","MER","New share"],y=["Revenue","Meta","Google","Other","Total media","Promos","MER","New share"],colorscale="RdBu",zmin=-1,zmax=1,text=np.round(cm.values,2),texttemplate="%{text}",hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
    plot_layout(fig,470)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Why the statistic matters</div>',unsafe_allow_html=True)
    explainers={
        "Pearson r":"Measures how closely two numeric variables follow a straight-line pattern. +1 means they rise together perfectly; −1 means one rises as the other falls; 0 means no linear pattern.",
        "Spearman ρ":"Looks at ranks instead of raw values. It is useful when the relationship is monotonic but not necessarily straight-line or when outliers are influential.",
        "p-value":"A small p-value means the observed relationship would be relatively unusual if the true population correlation were zero. It is not the probability that the relationship is causal.",
        "Anomaly score":"The dashboard uses a rolling median/MAD framework, which is more robust to extreme revenue spikes than a simple mean-and-standard-deviation z-score."
    }
    for title,text in explainers.items():
        with st.expander(title): st.write(text)

# ============================================================
# ANOMALY RADAR
# ============================================================
else:
    st.markdown('<div class="section">Anomaly Radar</div><div class="section-note">Find unusual weeks, inspect the surrounding business context and distinguish “unusual” from “explained”.</div>',unsafe_allow_html=True)
    threshold=st.slider("Robust anomaly threshold",2.0,5.0,3.0,0.5,key="anom_threshold")
    ad=d.copy(); ad["flag"]=ad.revenue_robust_z.abs()>=threshold
    st.markdown(f'<div class="callout"><b>Method:</b> rolling median + MAD. A week is flagged when |robust z| ≥ {threshold:.1f}. This is a screening tool, not an automatic explanation engine.</div>',unsafe_allow_html=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ad.date,y=ad.total_revenue,mode="lines+markers",name="Revenue",customdata=np.arange(len(ad)).reshape(-1,1),marker=dict(size=6)))
    flagged=ad[ad.flag]
    fig.add_trace(go.Scatter(x=flagged.date,y=flagged.total_revenue,mode="markers",name="Flagged anomaly",marker=dict(size=13,symbol="diamond",color=RED),customdata=flagged.index.to_numpy().reshape(-1,1)))
    interactive_chart(fig,ad,"anom_timeline","Anomaly week drill-down",500)

    if flagged.empty:
        st.success("No weeks cross the current threshold in the selected window.")
    else:
        st.markdown(f'<div class="section">{len(flagged)} flagged weeks</div>',unsafe_allow_html=True)
        show=flagged[["date","total_revenue","total_media_spend","MER","Promos","BFCM_promo","Price_Hike","revenue_robust_z"]].copy()
        show.columns=["Date","Revenue","Media spend","MER","Promos","BFCM","Price hike","Robust z"]
        event=st.dataframe(show,hide_index=True,use_container_width=True,on_select="rerun",selection_mode="single-row",key="anom_table")
        try:
            rows=event.selection.rows
            if rows:
                point_dialog(flagged.iloc[rows[0]],"Selected anomaly")
        except Exception: pass

    st.markdown('<div class="section">Investigation checklist</div>',unsafe_allow_html=True)
    for item in [
        "Was the week part of BFCM or another promotion-heavy period?",
        "Did media spend move unusually at the same time?",
        "Did new vs repeat customer mix shift?",
        "Did the price-hike indicator change?",
        "Does the same pattern persist in adjacent weeks?",
    ]:
        st.checkbox(item, key="check_"+str(abs(hash(item))))

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("SOLMAR Eyewear · Growth Intelligence · 109 weekly observations · Statistics-first, interactive decision support")
