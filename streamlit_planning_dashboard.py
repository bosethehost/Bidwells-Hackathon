# planning_risk_engine_excel.py
# Run: streamlit run planning_risk_engine_excel.py

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Config - path to uploaded excel
# ----------------------------
EXCEL_PATH = "data.xlsx"  # uploaded file

# ----------------------------
# Policy weight function (unchanged)
# ----------------------------
def calculate_policy_weights(planning_context: Dict[str, Any]) -> Dict[str, float]:
    weights = {"housing": 1.0, "brownfield": 1.0, "heritage": 1.0}
    if planning_context.get("five_year_supply") in ["No", "❌ No - Not demonstrated"]:
        weights["housing"] += 0.4
    elif planning_context.get("five_year_supply") in ["Marginal", "⚠️ Marginal"]:
        weights["housing"] += 0.2
    if planning_context.get("housing_delivery") in ["<75%", "Below 75%"]:
        weights["housing"] += 0.3
    elif planning_context.get("housing_delivery") in ["75-95%", "75–95%"]:
        weights["housing"] += 0.15
    if planning_context.get("local_plan_status") in ["Emerging", "Out-of-date", "📋 Emerging", "📜 Out-of-date"]:
        weights["housing"] += 0.2
    if planning_context.get("brownfield_policy") in ["Strong preference", "🎯 Strong preference", "Strong priority"]:
        weights["brownfield"] += 0.3
    elif planning_context.get("brownfield_policy") in ["Moderate preference", "⚖️ Moderate preference", "Moderate priority"]:
        weights["brownfield"] += 0.15
    if planning_context.get("heritage_context") in ["High sensitivity", "🏛️ High sensitivity", "High"]:
        weights["heritage"] += 0.4
    elif planning_context.get("heritage_context") in ["Moderate sensitivity", "⚖️ Moderate sensitivity", "Moderate"]:
        weights["heritage"] += 0.2
    for k in list(weights.keys()):
        weights[k] = min(max(weights[k], 0.5), 2.0)
    return weights

# ----------------------------
# Visual helpers (gauge + chart)
# ----------------------------
def create_score_gauge(score: int) -> go.Figure:
    if score >= 80:
        color = "#2ca02c"
    elif score >= 60:
        color = "#ff7f0e"
    elif score >= 40:
        color = "#d62728"
    else:
        color = "#8b0000"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Planning Success Score", 'font': {'size': 20}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 35], 'color': 'lightcoral'},
                {'range': [35, 70], 'color': 'lightyellow'},
                {'range': [70, 100], 'color': 'lightgreen'}],
        }))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_breakdown_chart(harms: List[Dict[str, Any]], benefits: List[Dict[str, Any]]) -> go.Figure:
    items = []
    for h in harms:
        items.append({"title": h["title"], "impact": -abs(h["impact"]), "type": "Risk"})
    for b in benefits:
        items.append({"title": b["title"], "impact": abs(b["impact"]), "type": "Benefit"})
    if not items:
        return None
    # top 4 by abs impact
    items_sorted = sorted(items, key=lambda x: abs(x["impact"]), reverse=True)[:4]
    df = pd.DataFrame(items_sorted)
    fig = px.bar(df, x="impact", y="title", color="type", orientation="h",
                 color_discrete_map={"Risk": "#d62728", "Benefit": "#2ca02c"},
                 title="Top 4 Risk/Benefit Impacts")
    fig.update_layout(height=360, yaxis={'categoryorder': 'total ascending'})
    return fig

# ----------------------------
# Load Excel containing constraints (0-1)
# ----------------------------
@st.cache_data(ttl=600)
def load_constraints(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name=0)
    # first column contains scenario names (Commercial / Residential)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # ensure first column is scenario
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = [str(i).strip() for i in df.index]
    return df

# ----------------------------
# Mapping which columns count as benefits vs harms
# (Edit if you want different semantics)
# ----------------------------
IS_BENEFIT_COLUMN = {
    "Green Belt": False,
    "Brownfield": True,
    "Heritage Site?": False,
    "Known Contamination Risk?": False,
    "Minimum 10% Biodiversity Net Gain": True,
    "Known Flood Risk": False,
    "Conservation Area": False,
    "High Levels of Air pollution": False,
    "High Levels of Noise pollution": False,
    "Sufficient Utility Capacity": True,
    "Sufficient Transport Connectivity": True,
    "Protected Employment Land": False,
    "Density compliance": True,
    "Sufficient Total Housing": True,
    "35% Affordable Housing": True,
    "Aesthetical alginment": True,
    "Sufficent Housing mixture": True
}

COLUMN_DESCRIPTIONS = {
    "Green Belt": "Site lies in the Green Belt.",
    "Brownfield": "Site is previously developed (brownfield).",
    "Heritage Site?": "Close to or within setting of a heritage asset.",
    "Known Contamination Risk?": "Known/suspected contamination.",
    "Minimum 10% Biodiversity Net Gain": "Meets biodiversity net gain expectations.",
    "Known Flood Risk": "Flood risk present.",
    "Conservation Area": "Within or adjacent to a conservation area.",
    "High Levels of Air pollution": "High local air pollution.",
    "High Levels of Noise pollution": "High local noise levels.",
    "Sufficient Utility Capacity": "Sufficient utility capacity.",
    "Sufficient Transport Connectivity": "Good transport connections.",
    "Protected Employment Land": "Designated protected employment land.",
    "Density compliance": "Density is compliant with local expectations.",
    "Sufficient Total Housing": "Sufficient housing totals locally.",
    "35% Affordable Housing": "35% affordable housing provision.",
    "Aesthetical alginment": "Design aligns with local character.",
    "Sufficent Housing mixture": "Appropriate housing mix."
}

IMPACT_MULTIPLIER = 10.0  # excel 0-1 -> impact 0-10

# ----------------------------
# auto_assess_site using excel constraints
# ----------------------------
def auto_assess_site(site_meta: Dict[str, Any],
                     constraints_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    site_meta: from session_state with keys:
      - primary_use: Residential/Commercial/Mixed
      - dwellings (int)
      - total_floorspace (float)
      - percent_res (0-100) if mixed
      - planning_context (optional)
    constraints_df: dataframe indexed by scenario names (e.g., 'Commercial', 'Residential')
    """
    primary_use = site_meta.get("primary_use", "Residential")
    pct_res = float(site_meta.get("percent_residential", 100)) / 100.0 if primary_use == "Mixed" else (1.0 if primary_use == "Residential" else 0.0)
    pct_com = 1.0 - pct_res

    # attempt to find rows - commonly 'Residential' and 'Commercial' exist in your sheet
    # handle trailing whitespace / tabs
    def fetch_row(name_candidates):
        for n in name_candidates:
            if n in constraints_df.index:
                return constraints_df.loc[n].astype(float)
            # try trimmed/lower
            for idx in constraints_df.index:
                if str(idx).strip().lower() == str(n).strip().lower():
                    return constraints_df.loc[idx].astype(float)
        return pd.Series(dtype=float)

    res_row = fetch_row(["Residential", "EC Resid", "Resid"])
    com_row = fetch_row(["Commercial", "EC Comm", "Comm", "Commercial\t"])

    # determine union of columns
    cols = sorted(set(list(res_row.index) + list(com_row.index)))

    # compute combined constraint values (0-1) depending on primary use
    combined = {}
    for c in cols:
        rv = float(res_row.get(c, 0.0)) if not res_row.empty else 0.0
        cv = float(com_row.get(c, 0.0)) if not com_row.empty else 0.0
        if primary_use == "Residential":
            combined[c] = rv
        elif primary_use == "Commercial":
            combined[c] = cv
        else:  # Mixed
            # if constraint missing in one row, that value is 0 (as requested)
            combined[c] = rv * pct_res + cv * pct_com

    # Build harms and benefits from combined dict
    harms = []
    benefits = []

    # Convert constraints (0-1) to impacts (0-10)
    for c, v in combined.items():
        if pd.isna(v) or v == 0:
            continue
        impact = round(float(v) * IMPACT_MULTIPLIER, 2)
        title = c
        desc = COLUMN_DESCRIPTIONS.get(c, f"{c}: score {v:.2f}")
        if IS_BENEFIT_COLUMN.get(c, False):
            benefits.append({"title": title, "desc": desc, "impact": impact})
        else:
            harms.append({"title": title, "desc": desc, "impact": impact, "mitigation": None})

    # Add housing / employment benefits using dwellings and floorspace (as before)
    dwellings = int(site_meta.get("dwellings", 0))
    floorspace = float(site_meta.get("total_floorspace", 0.0))
    # simple normalized scores
    def norm_dwell(x):
        try:
            return 1.0 / (1.0 + math.exp(-(x - 100) / 75.0))
        except:
            return 0.0
    def norm_floor(x):
        if x <= 0: return 0.0
        return min(1.0, math.log1p(x) / math.log(50000.0))
    ds = norm_dwell(dwellings)
    fs = norm_floor(floorspace)
    if primary_use in ("Residential", "Mixed"):
        benefits.append({"title": "Housing delivery", "desc": "Scheme delivers housing to local market", "impact": round(ds * 8.0 + fs * 2.0, 2)})
    if primary_use in ("Commercial", "Mixed"):
        benefits.append({"title": "Employment / commercial floorspace", "desc": "Scheme provides commercial floorspace and potential jobs", "impact": round(fs * 8.0 + ds * 1.0, 2)})

    # defaults
    if not harms:
        harms.append({"title": "Low obvious policy conflict", "desc": "No high-level constraints detected.", "impact": 1, "mitigation": None})
    if not benefits:
        benefits.append({"title": "Development potential", "desc": "Proposal would deliver development & local benefits.", "impact": 2})

    # Add mitigation templates where appropriate (simple mapping)
    for h in harms:
        t = h["title"].lower()
        if "flood" in t:
            h["mitigation"] = "Sequential test; raise finished floor levels; SuDS; safe access/egress; FRA."
        elif "heritage" in t or "conservation" in t:
            h["mitigation"] = "Heritage statement, sensitive design, materials, and setting protection."
        elif "green belt" in t:
            h["mitigation"] = "Demonstrate Very Special Circumstances (VSC); consider brownfield reuse and alternatives."
        elif "contamination" in t:
            h["mitigation"] = "Site investigation, remediation, monitoring and verification plan."
        elif "air" in t or "noise" in t:
            h["mitigation"] = "AQ & noise assessments, mitigation (acoustic glazing, ventilation), layout changes."
        elif "protected employment" in t:
            h["mitigation"] = "Engage the council, justify change of use or provide employment retention strategy."
        else:
            if h.get("mitigation") is None:
                h["mitigation"] = "Prepare constraint-specific mitigation and document in submission."

    return harms, benefits

# ----------------------------
# Balance engine (same as yours)
# ----------------------------
def planning_balance_engine(harms: List[Dict[str, Any]],
                            benefits: List[Dict[str, Any]],
                            local_policy_priority: Dict[str, float] = None) -> Dict[str, Any]:
    harm_score = sum(h.get("impact", 0) for h in harms)
    benefit_score = sum(b.get("impact", 0) for b in benefits)

    # Apply policy weights to benefits that mention keys
    if local_policy_priority:
        for benefit in benefits:
            for key, weight in local_policy_priority.items():
                if key.lower() in benefit.get("title", "").lower() or key.lower() in benefit.get("desc","").lower():
                    benefit_score += benefit.get("impact", 0) * (weight - 1.0)

    raw = (benefit_score - harm_score) + 20
    mapped = 1 / (1 + math.exp(-(raw / 15)))
    score = int(round(mapped * 100))
    score = max(0, min(100, score))
    if score >= 80:
        label = "Low Risk / Likely to Succeed"; icon = "🟢"
    elif score >= 60:
        label = "Medium Risk / Reasonable Chance"; icon = "🟡"
    elif score >= 40:
        label = "High Risk / Uncertain"; icon = "🟠"
    else:
        label = "Very High Risk / Unlikely"; icon = "🔴"
    harms_sorted = sorted(harms, key=lambda x: -abs(x.get("impact", 0)))[:3]
    benefits_sorted = sorted(benefits, key=lambda x: -abs(x.get("impact", 0)))[:3]
    return {"score": score, "label": label, "icon": icon,
            "rationale": {"harm_score": harm_score, "benefit_score": benefit_score,
                          "top_harms": harms_sorted, "top_benefits": benefits_sorted}}

# ----------------------------
# Mitigation paragraph generator (top 2 harms)
# ----------------------------
def mitigation_text_for_harm(harm: Dict[str, Any], site_meta: Dict[str, Any]) -> str:
    t = harm.get("title","")
    desc = harm.get("desc","")
    mitigation = harm.get("mitigation","Develop a mitigation plan.")
    txt = f"**{t}:** {desc}\n\n{mitigation}\n\n"
    # add site-specific tailoring
    if "housing" in site_meta.get("planning_context", {}).get("housing_delivery","").lower():
        txt += ""
    return txt

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Planning Risk Engine — Excel-driven", layout="wide", page_icon="🏗️")
    st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px; gap: 8px; padding-top:10px; padding-bottom:10px }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏗️ Planning Risk Engine (Excel-driven)")

    # load constraints
    constraints_df = load_constraints(EXCEL_PATH)
    if constraints_df.empty:
        st.error(f"Could not load constraints Excel at {EXCEL_PATH}. Please check the file path and sheet format.")
        st.stop()

    # session init
    if 'site_meta' not in st.session_state:
        st.session_state.site_meta = {
            "site_choice": "Elephant and Cast",
            "primary_use": "Residential",
            "dwellings": 120,
            "total_floorspace": 5000,
            "percent_residential": 100,
            "planning_context": {
                "five_year_supply": "Yes",
                "housing_delivery": ">95%",
                "local_plan_status": "Adopted (<5 years)",
                "brownfield_policy": "Strong preference",
                "heritage_context": "High sensitivity"
            }
        }
        st.session_state.policy_weights = calculate_policy_weights(st.session_state.site_meta["planning_context"])
        st.session_state.current_assessment = None

    # Sidebar: simplified as requested
    with st.sidebar:
        st.header("🎯 Scenario Configuration")
        site_choice = st.selectbox("Scenario Name", ["Elephant and Cast"])  # only option as requested
        primary_use = st.selectbox("Primary Purpose", ["Residential", "Commercial", "Mixed"])
        dwellings = st.number_input("Number of Dwellings", min_value=0, value=int(st.session_state.site_meta.get("dwellings",120)))
        total_floorspace = st.number_input("Total Floorspace (m²)", min_value=0, value=int(st.session_state.site_meta.get("total_floorspace",5000)))
        percent_residential = 100
        if primary_use == "Mixed":
            percent_residential = st.slider("% Residential (Mixed)", 0, 100, 50)

        # Minimal quick policy context used to compute policy_weights (keeps previous behaviour)
        st.markdown("---")
        st.subheader("Local Planning Context (quick)")
        five_year_supply = st.selectbox("5-year housing land supply", ["✅ Yes - Demonstrated", "⚠️ Marginal", "❌ No - Not demonstrated"], index=0)
        housing_delivery = st.selectbox("Housing delivery rate", [">95%", "75-95%", "<75%"], index=0)
        local_plan_status = st.selectbox("Local plan status", ["📅 Adopted (<5 years)", "📋 Emerging", "📜 Out-of-date"], index=0)

        if st.button("💾 Save Configuration", use_container_width=True):
            st.session_state.site_meta = {
                "site_choice": site_choice,
                "primary_use": primary_use,
                "dwellings": dwellings,
                "total_floorspace": total_floorspace,
                "percent_residential": percent_residential,
                "planning_context": {
                    "five_year_supply": five_year_supply,
                    "housing_delivery": housing_delivery,
                    "local_plan_status": local_plan_status
                }
            }
            st.session_state.policy_weights = calculate_policy_weights(st.session_state.site_meta["planning_context"])
            st.success("Configuration saved!")

    # Tabs (same names as original)
    tab1, tab2 = st.tabs(["📊 Risk Assessment", "📈 Results & Analysis"])

    # Tab 1: Risk Assessment
    with tab1:
        st.header("Risk Assessment")
        col1, col2 = st.columns([3,1])
        with col1:
            meta = st.session_state.site_meta
            st.subheader("Current Scenario")
            st.write(f"**Site:** {meta.get('site_choice','Elephant and Cast')}")
            st.write(f"**Use:** {meta.get('primary_use')}")
            st.write(f"**Dwellings:** {meta.get('dwellings')}")
            st.write(f"**Floorspace:** {meta.get('total_floorspace')}")
            st.subheader("📊 Calculated Policy Weights")
            if 'policy_weights' in st.session_state:
                w = st.session_state.policy_weights
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Housing Weight", f"{w['housing']:.2f}")
                with c2: st.metric("Brownfield Weight", f"{w['brownfield']:.2f}")
                with c3: st.metric("Heritage Weight", f"{w['heritage']:.2f}")
        with col2:
            if st.button("🚀 Run Assessment", use_container_width=True):
                with st.spinner("Running assessment (Excel-driven)..."):
                    harms, benefits = auto_assess_site(st.session_state.site_meta, constraints_df)
                    balance = planning_balance_engine(harms, benefits, st.session_state.policy_weights)
                    st.session_state.current_assessment = {"harms": harms, "benefits": benefits, "balance": balance}

        if st.session_state.current_assessment:
            bal = st.session_state.current_assessment["balance"]
            st.plotly_chart(create_score_gauge(bal['score']), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Overall Score", f"{bal['score']}%")
            with c2: st.metric("Benefit Score", f"{bal['rationale']['benefit_score']:.1f}")
            with c3: st.metric("Risk Score", f"{bal['rationale']['harm_score']:.1f}")
            st.write(f"**Verdict:** {bal['icon']} {bal['label']}")
        else:
            st.info("Save configuration and click 'Run Assessment' to analyze your scenario.")

    # Tab 2: Results & Analysis
    with tab2:
        st.header("Detailed Results & Analysis")
        if not st.session_state.current_assessment:
            st.info("Run an assessment in the Risk Assessment tab first.")
        else:
            assessment = st.session_state.current_assessment
            harms = assessment["harms"]
            benefits = assessment["benefits"]
            balance = assessment["balance"]

            st.subheader("Risk-Benefit Analysis (Top 4)")
            risk_chart = create_risk_breakdown_chart(harms, benefits)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Key Benefits (Top 3)")
                for b in balance["rationale"]["top_benefits"]:
                    st.info(f"**{b['title']}**")
                    st.write(b.get("desc",""))
                    st.caption(f"Impact: +{b['impact']}")
            with col2:
                st.subheader("❌ Key Risks (Top 3)")
                for h in balance["rationale"]["top_harms"]:
                    st.error(f"**{h['title']}**")
                    st.write(h.get("desc",""))
                    st.caption(f"Impact: {h['impact']}")
                    if h.get("mitigation"):
                        st.write(f"*Mitigation:* {h['mitigation']}")

            st.markdown("---")
            st.subheader("🎯 Top 2 Risk Mitigation Recommendations")
            top_harms = sorted(balance["rationale"]["top_harms"], key=lambda x: x.get("impact",0), reverse=True)[:2]
            if not top_harms:
                st.info("No significant harms to provide recommendations for.")
            else:
                recs = ""
                for h in top_harms:
                    recs += mitigation_text_for_harm(h, st.session_state.site_meta)
                st.markdown(recs)

            with st.expander("Debug: full harms & benefits"):
                st.write("Combined constraints used (excerpt):")
                # show combined constraints used
                # we can reconstruct combined if necessary, but show extracted arrays
                st.json({"harms": harms, "benefits": benefits, "balance": balance})

if __name__ == "__main__":
    main()



