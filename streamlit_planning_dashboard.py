import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import hashlib
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 
# Automatic Policy Weight Calculation
# -------------------------------

def calculate_policy_weights(planning_context: Dict[str, Any]) -> Dict[str, float]:
    """Calculate policy weights based on actual planning context that planners know"""
    weights = {
        "housing": 1.0,
        "brownfield": 1.0,
        "heritage": 1.0
    }
    
    # Housing weight based on supply and delivery
    if planning_context.get("five_year_supply") == "No":
        weights["housing"] += 0.4
    elif planning_context.get("five_year_supply") == "Marginal":
        weights["housing"] += 0.2
        
    if planning_context.get("housing_delivery") == "<75%":
        weights["housing"] += 0.3
    elif planning_context.get("housing_delivery") == "75-95%":
        weights["housing"] += 0.15
        
    if planning_context.get("local_plan_status") in ["Emerging", "Out-of-date"]:
        weights["housing"] += 0.2
    
    # Brownfield weight based on local policy
    if planning_context.get("brownfield_policy") == "Strong preference":
        weights["brownfield"] += 0.3
    elif planning_context.get("brownfield_policy") == "Moderate preference":
        weights["brownfield"] += 0.15
        
    # Heritage weight based on sensitivity
    if planning_context.get("heritage_context") == "High sensitivity":
        weights["heritage"] += 0.4
    elif planning_context.get("heritage_context") == "Moderate sensitivity":
        weights["heritage"] += 0.2
    
    # Cap weights at reasonable limits
    for key in weights:
        weights[key] = min(weights[key], 2.0)
        weights[key] = max(weights[key], 0.5)
    
    return weights

# -------------------------------
# Visualization Functions
# -------------------------------

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
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Planning Success Score", 'font': {'size': 20}},
        number = {'font': {'size': 40}},
        gauge = {
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
    if not harms and not benefits:
        return None
        
    data = []
    for harm in harms:
        data.append({'title': harm.get('title', ''), 'impact': harm.get('impact', 0), 'type': 'Risk'})
    for benefit in benefits:
        data.append({'title': benefit.get('title', ''), 'impact': benefit.get('impact', 0), 'type': 'Benefit'})
    
    if not data:
        return None
        
    df = pd.DataFrame(data)
    df['abs_impact'] = df['impact'].abs()
    df = df.sort_values('abs_impact', ascending=True)
    
    fig = px.bar(df, x='impact', y='title', color='type',
                 orientation='h',
                 color_discrete_map={'Risk': '#d62728', 'Benefit': '#2ca02c'},
                 title="Risk/Benefit Impact Analysis")
    
    fig.update_layout(height=max(250, len(data) * 25), showlegend=True)
    return fig

# -------------------------------
# Core Engine Functions
# -------------------------------

@st.cache_data(ttl=3600)
def auto_assess_site(site_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    harms = []
    benefits = []
    
    # Flood risk
    flood_zone = site_meta.get("flood_zone")
    if flood_zone in ("2", "3"):
        harms.append({"title": "Flood risk", "desc": f"Site lies in Flood Zone {flood_zone}.", "impact": 8, "mitigation": "Sequential test; raise finished floor levels; SuDS."})
    
    # Heritage
    if site_meta.get("adjacent_heritage"):
        harms.append({"title": "Heritage asset adjacency", "desc": "Close to listed building or conservation area.", "impact": 7, "mitigation": "Heritage Impact Assessment; avoid harm to setting."})
    
    # Green Belt
    if site_meta.get("green_belt"):
        harms.append({"title": "Green Belt", "desc": "Site is in the Green Belt.", "impact": 10, "mitigation": "Very special circumstances required."})
    
    # Brownfield benefits/risks
    if site_meta.get("brownfield"):
        if site_meta.get("contamination_risk"):
            harms.append({"title": "Contamination risk", "desc": "Likely to require remediation works.", "impact": 6, "mitigation": "Site investigations and remediation."})
        benefits.append({"title": "Brownfield regeneration", "desc": "Re-use of previously developed land supports NPPF objectives.", "impact": 6})
    
    # Housing need - now based on actual planning context
    if site_meta.get("five_year_supply") in ["No", "Marginal"]:
        benefits.append({"title": "Housing delivery", "desc": "Local authority failing 5YHLS or high housing need.", "impact": 8})
    
    # Transport/access
    if not site_meta.get("good_access"):
        harms.append({"title": "Poor access / highways impact", "desc": "Limited access or constrained highway network.", "impact": 5, "mitigation": "Transport Assessment and junction mitigation."})
    else:
        benefits.append({"title": "Good access to public transport", "desc": "Close to public transport links.", "impact": 3})
    
    # Affordable housing
    if site_meta.get("affordable_pct", 0) >= 25:
        benefits.append({"title": "Affordable housing offered", "desc": f"{site_meta.get('affordable_pct')}% affordable homes.", "impact": 5})
    
    # Defaults
    if not harms:
        harms.append({"title": "Low obvious policy conflict", "desc": "No high-level constraints detected.", "impact": 1})
    if not benefits:
        benefits.append({"title": "Development potential", "desc": "Proposal would deliver development & local benefits.", "impact": 2})
    
    return harms, benefits

@st.cache_data(ttl=3600)
def planning_balance_engine(harms: List[Dict[str, Any]],
                          benefits: List[Dict[str, Any]],
                          local_policy_priority: Dict[str, float] = None) -> Dict[str, Any]:
    
    harm_score = sum(h.get("impact", 0) for h in harms)
    benefit_score = sum(b.get("impact", 0) for b in benefits)
    
    # Apply policy weights
    if local_policy_priority:
        for benefit in benefits:
            for key, weight in local_policy_priority.items():
                if key in benefit.get("title", "").lower():
                    benefit_score += benefit.get("impact", 0) * (weight - 1.0)
    
    raw = (benefit_score - harm_score) + 20
    mapped = 1 / (1 + math.exp(-(raw / 15)))
    score = int(round(mapped * 100))
    score = max(0, min(100, score))
    
    if score >= 80:
        label = "Low Risk / Likely to Succeed"
        icon = "🟢"
    elif score >= 60:
        label = "Medium Risk / Reasonable Chance"
        icon = "🟡"
    elif score >= 40:
        label = "High Risk / Uncertain"
        icon = "🟠"
    else:
        label = "Very High Risk / Unlikely"
        icon = "🔴"
    
    harms_sorted = sorted(harms, key=lambda x: -abs(x.get("impact", 0)))[:3]
    benefits_sorted = sorted(benefits, key=lambda x: -abs(x.get("impact", 0)))[:3]
    
    return {
        "score": score, 
        "label": label, 
        "icon": icon, 
        "rationale": {
            "harm_score": harm_score,
            "benefit_score": benefit_score,
            "top_harms": harms_sorted,
            "top_benefits": benefits_sorted
        }
    }

# -------------------------------
# Main Application with Organized Sidebar
# -------------------------------

def main():
    st.set_page_config(
        page_title="Planning Risk Engine", 
        layout="wide",
        page_icon="🏗️"
    )
    
    # Custom CSS for cleaner look
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏗️ Planning Risk Engine")
    
    # Initialize session state
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    
    # =============================
    # SIDEBAR - Planner-Friendly Inputs
    # =============================
    with st.sidebar:
        st.header("🎯 Scenario Configuration")
        
        # Scenario Basics
        st.subheader("Basic Information")
        site_choice = st.text_input("Scenario Name", value="Triton Square")
        primary_use = st.selectbox("Primary Use", ["Residential", "Commercial", "Mixed"])
        dwellings = st.number_input("Number of Dwellings", min_value=0, value=120)
        affordable_pct = st.slider("Affordable Housing (%)", 0, 100, 25)
        total_floorspace = st.number_input("Total Floorspace (m²)", min_value=0, value=5000)
        
        # Site Constraints
        st.subheader("📍 Site Constraints")
        brownfield = st.checkbox("Brownfield Site", value=True)
        contamination_risk = st.checkbox("Contamination Risk", value=False)
        green_belt = st.checkbox("In Green Belt", value=False)
        adjacent_heritage = st.checkbox("Adjacent to Heritage Asset", value=True)
        flood_zone = st.selectbox("Flood Zone", ["None", "1", "2", "3"])
        good_access = st.checkbox("Good Public Transport Access", value=True)
        
        # Planning Policy Context (What planners ACTUALLY know)
        st.subheader("🏛️ Local Planning Context")
        five_year_supply = st.radio("5-year housing land supply", 
                                  ["✅ Yes - Demonstrated", "⚠️ Marginal", "❌ No - Not demonstrated"],
                                  help="Critical for tilted balance considerations")
        
        housing_delivery = st.selectbox("Housing delivery rate",
                                      [">95%", "75-95%", "<75%"],
                                      help="Percentage of housing target actually delivered")
        
        local_plan_status = st.selectbox("Local plan status",
                                       ["📅 Adopted (<5 years)", "📋 Emerging", "📜 Out-of-date"],
                                       help="How current are local policies")
        
        brownfield_policy = st.radio("Council brownfield policy",
                                   ["🎯 Strong preference", "⚖️ Moderate preference", "📈 No specific policy"])
        
        heritage_context = st.selectbox("Local heritage sensitivity",
                                      ["🏛️ High sensitivity", "⚖️ Moderate sensitivity", "📊 Low sensitivity"])
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        if st.button("💾 Save Current Configuration", use_container_width=True):
            planning_context = {
                "five_year_supply": five_year_supply,
                "housing_delivery": housing_delivery,
                "local_plan_status": local_plan_status,
                "brownfield_policy": brownfield_policy,
                "heritage_context": heritage_context
            }
            
            # Calculate policy weights automatically
            policy_weights = calculate_policy_weights(planning_context)
            
            st.session_state.site_meta = {
                "site_choice": site_choice,
                "primary_use": primary_use,
                "dwellings": dwellings,
                "affordable_pct": affordable_pct,
                "total_floorspace": total_floorspace,
                "brownfield": brownfield,
                "contamination_risk": contamination_risk,
                "green_belt": green_belt,
                "adjacent_heritage": adjacent_heritage,
                "flood_zone": flood_zone if flood_zone != "None" else None,
                "good_access": good_access,
                "five_year_supply": five_year_supply,
                "housing_delivery": housing_delivery,
                "local_plan_status": local_plan_status,
            }
            st.session_state.policy_weights = policy_weights
            st.session_state.planning_context = planning_context
            st.success("Configuration saved!")
        
        # System Info
        st.markdown("---")
        st.caption("ℹ️ Configure your scenario in the sidebar, then use the tabs above for analysis")
    
    # Initialize session state if not exists
    if 'site_meta' not in st.session_state:
        planning_context = {
            "five_year_supply": "✅ Yes - Demonstrated",
            "housing_delivery": ">95%",
            "local_plan_status": "📅 Adopted (<5 years)",
            "brownfield_policy": "🎯 Strong preference",
            "heritage_context": "🏛️ High sensitivity"
        }
        
        st.session_state.site_meta = {
            "site_choice": site_choice,
            "primary_use": primary_use,
            "dwellings": dwellings,
            "affordable_pct": affordable_pct,
            "total_floorspace": total_floorspace,
            "brownfield": brownfield,
            "contamination_risk": contamination_risk,
            "green_belt": green_belt,
            "adjacent_heritage": adjacent_heritage,
            "flood_zone": flood_zone if flood_zone != "None" else None,
            "good_access": good_access,
            "five_year_supply": "✅ Yes - Demonstrated",
            "housing_delivery": ">95%",
            "local_plan_status": "📅 Adopted (<5 years)",
        }
        st.session_state.policy_weights = calculate_policy_weights(planning_context)
        st.session_state.planning_context = planning_context
    
    # =============================
    # MAIN CONTENT TABS
    # =============================
    tab1, tab3 = st.tabs([
        "📊 Risk Assessment",
        "📈 Results & Analysis"
    ])
    
    # Tab 1: Risk Assessment
    with tab1:
        st.header("Risk Assessment")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Current Scenario")
            st.write(f"**Site:** {st.session_state.site_meta['site_choice']}")
            st.write(f"**Use:** {st.session_state.site_meta['primary_use']}")
            st.write(f"**Dwellings:** {st.session_state.site_meta['dwellings']}")
            st.write(f"**Affordable Housing:** {st.session_state.site_meta['affordable_pct']}%")
            
            # Show calculated policy weights
            st.subheader("📊 Calculated Policy Weights")
            if 'policy_weights' in st.session_state:
                weights = st.session_state.policy_weights
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    st.metric("Housing Weight", f"{weights['housing']:.2f}")
                with col_w2:
                    st.metric("Brownfield Weight", f"{weights['brownfield']:.2f}")
                with col_w3:
                    st.metric("Heritage Weight", f"{weights['heritage']:.2f}")
        
        with col2:
            if st.button("🚀 Run Assessment", type="primary", use_container_width=True):
                with st.spinner("Analyzing site constraints and benefits..."):
                    harms, benefits = auto_assess_site(st.session_state.site_meta)
                    balance = planning_balance_engine(harms, benefits, st.session_state.policy_weights)
                    
                    st.session_state.current_assessment = {
                        "harms": harms,
                        "benefits": benefits,
                        "balance": balance
                    }
        
        # FIX: Check if assessment exists before accessing it
        if st.session_state.current_assessment is not None:
            assessment = st.session_state.current_assessment
            balance = assessment["balance"]
            
            # Score Display
            st.plotly_chart(create_score_gauge(balance['score']), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{balance['score']}%")
            with col2:
                st.metric("Benefit Score", f"{balance['rationale']['benefit_score']:.1f}")
            with col3:
                st.metric("Risk Score", f"{balance['rationale']['harm_score']:.1f}")
            
            st.write(f"**Verdict:** {balance['icon']} {balance['label']}")
        else:
            st.info("Click 'Run Assessment' to analyze your scenario")
    
    # Tab 3: Results & Analysis
    with tab3:
        st.header("Detailed Results & Analysis")
        
        if st.session_state.current_assessment is None:
            st.info("Run an assessment in the Risk Assessment tab first to see detailed results.")
        else:
            assessment = st.session_state.current_assessment
            balance = assessment["balance"]
            harms = assessment["harms"]
            benefits = assessment["benefits"]
            
            # Risk/Benefit Chart
            st.subheader("Risk-Benefit Analysis")
            risk_chart = create_risk_breakdown_chart(harms, benefits)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Detailed Breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Key Benefits")
                for benefit in balance["rationale"]["top_benefits"]:
                    with st.container():
                        st.info(f"**{benefit['title']}**")
                        st.write(benefit['desc'])
                        st.caption(f"Impact: +{benefit['impact']}")
            
            with col2:
                st.subheader("❌ Key Risks")
                for harm in balance["rationale"]["top_harms"]:
                    with st.container():
                        st.error(f"**{harm['title']}**")
                        st.write(harm['desc'])
                        st.caption(f"Impact: {harm['impact']}")
                        if harm.get('mitigation'):
                            st.write(f"*Mitigation: {harm['mitigation']}*")
            
            # Recommendations
            st.subheader("🎯 Recommendations")

        if st.session_state.current_assessment:
        
            harms = st.session_state.current_assessment["balance"]["rationale"]["top_harms"]
            benefits = st.session_state.current_assessment["balance"]["rationale"]["top_benefits"]
            policy_weights = st.session_state.policy_weights
        
            report = ""
        
            # --- Benefits narrative ---
            if benefits:
                report += "The proposed development offers several significant benefits that should be emphasized in the planning submission. "
                for b in benefits:
                    if "housing" in b['title'].lower():
                        report += (
                            "Given the shortfall in housing delivery in the local area, this scheme can make an important contribution to meeting local housing needs. "
                            "It is essential to clearly demonstrate how the proposal will deliver both market and affordable housing, with evidence of realistic delivery timelines and mechanisms. "
                            "Policy-aligned benefits should be highlighted in the planning statement to support any tilted-balance considerations under national policy frameworks. "
                        )
                    elif "brownfield" in b['title'].lower():
                        report += (
                            "The redevelopment of previously developed land provides clear environmental and sustainability benefits. "
                            "Brownfield regeneration reduces pressure on greenfield or green belt sites and aligns strongly with national and local planning policies promoting efficient land use. "
                            "A clear statement of the environmental improvements, remediation works, and wider community benefits should be incorporated into the submission. "
                        )
                    elif "heritage" in b['title'].lower():
                        report += (
                            "The scheme has the opportunity to enhance the setting of adjacent heritage assets through sensitive design. "
                            "Design statements should highlight how the development complements the historic environment while providing modern amenities, with careful consideration of materials, massing, and landscaping. "
                        )
                    else:
                        report += f"{b['title']} is a positive aspect of the scheme and should be clearly documented in the planning narrative. "
        
            # --- Harms narrative with detailed planning guidance ---
            if harms:
                report += "Several risks have been identified that require careful mitigation and justification. "
                for h in harms:
                    impact = h.get("impact", 0)
        
                    if "flood" in h['title'].lower():
                        report += (
                            "Given the significant flood risk identified, the first priority must be to commission a site-specific Flood Risk Assessment (FRA) in line with national planning policy. "
                            "The FRA should demonstrate not only how the development will remain safe over its lifetime but also that it will not increase flood risk elsewhere. "
                            "It should incorporate climate change allowances, and early engagement with the Environment Agency and Lead Local Flood Authorities is strongly recommended to ensure that the proposed mitigation strategy is robust and acceptable. "
                        )
                    elif "heritage" in h['title'].lower():
                        report += (
                            "On the heritage impact front, the proximity to a listed building or conservation area necessitates a detailed heritage statement. "
                            "This should assess the significance of any impacted asset, the contribution of the site to its setting, and propose design-led mitigation such as careful massing, landscaping, or choice of materials to reduce visual or experiential harm. "
                            "Including this as part of a broader planning / design and access statement will justify how the scheme responds to policy while preserving historic character. "
                        )
                    elif "green belt" in h['title'].lower():
                        report += (
                            "As the site lies within or adjacent to the Green Belt, any development will need to demonstrate very special circumstances (VSC) to justify any inappropriate development. "
                            "The planning submission should include a detailed Green Belt statement outlining the harm to openness and permanence, alongside quantified benefits such as housing delivery, brownfield regeneration, and community enhancements. "
                            "The VSC argument must show that public benefits clearly outweigh any harm to the Green Belt. "
                        )
                    elif "contamination" in h['title'].lower():
                        report += (
                            "Where contamination risks are present, commissioning thorough ground investigations and remediation strategies is essential. "
                            "The planning submission should include details of the remediation approach, monitoring plans, and verification processes to assure the local authority of the site's safety and environmental compliance. "
                        )
                    elif "access" in h['title'].lower():
                        report += (
                            "Constraints on access or highways require a robust Transport Assessment. "
                            "The assessment should identify junction improvements, trip generation, and pedestrian/cyclist connectivity. "
                            "Mitigation measures must be clearly defined and supported by technical evidence to ensure compliance with local and national highway standards. "
                        )
                    else:
                        report += f"{h['title']} is a noted risk that should be assessed and mitigated with appropriate supporting documentation. "
        
            # --- Strategic synthesis paragraph --- 
            st.markdown(report)
        
        else:
            st.info("Run an assessment first to generate recommendations.")

if __name__ == "__main__":
    main()




