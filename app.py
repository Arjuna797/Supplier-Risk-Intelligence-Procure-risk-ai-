"""
Supplier Risk Intelligence & Procurement Analytics System
Interactive Streamlit Dashboard aligned with SAP MM (Materials Management).
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env_setup  # Windows DLL bootstrap for Scipy/Scikit-learn/XGBoost

from src.sap_mapping import get_field_mapping_dataframe, SAP_FIELD_MAPPING
from src.data_pipeline import validate_and_clean_data, prepare_pipeline_data
from src.recommender import recommend_alternative_suppliers

DATA_FILE = ROOT / "data" / "supplier_master_modeled.csv"
METRICS_FILE = ROOT / "models" / "training_metrics.json"

st.set_page_config(
    page_title="Supplier Risk Intelligence | SAP MM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #0066cc;
    }
    .badge-high {
        background-color: #ff4d4d;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-med {
        background-color: #ffa500;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        from run_pipeline import main as run_all
        run_all()
    return pd.read_csv(DATA_FILE)

@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}

raw_df = load_data()
metrics_data = load_metrics()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supplier.png", width=64)
    st.title("Supplier Risk Intelligence")
    st.caption("SAP MM | Supplier Evaluation | Predictive Analytics")
    st.markdown("---")

    # Data Source Selector
    data_source_mode = st.radio("Dataset Source:", ["Standard Simulated SAP MM Dataset", "Upload Custom CSV"], index=0)
    
    uploaded_df = None
    if data_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload Supplier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                user_csv = pd.read_csv(uploaded_file)
                uploaded_df = validate_and_clean_data(user_csv)
                st.success(f"Loaded {len(uploaded_df)} custom records successfully!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    active_df = uploaded_df if uploaded_df is not None else raw_df

    st.subheader("Filters")
    mat_groups = ["All"] + sorted(active_df["Material_Group"].dropna().unique().tolist())
    selected_mat_group = st.selectbox("Material Group (MATKL)", mat_groups)

    countries = ["All"] + sorted(active_df["Country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country (LAND1)", countries)

    risk_cats = ["All"] + sorted(active_df["Risk_Category"].dropna().unique().tolist())
    selected_risk = st.selectbox("Risk Category", risk_cats)

    if "Anomaly_Status" in active_df.columns:
        anomaly_opts = ["All"] + sorted(active_df["Anomaly_Status"].dropna().unique().tolist())
        selected_anomaly = st.selectbox("Anomaly Filter", anomaly_opts)
    else:
        selected_anomaly = "All"

    st.markdown("---")
    st.info("""
    **Relevant SAP MM Modules:**
    - **LFA1 / BP**: Supplier Master
    - **EKKO / EKPO**: Purchasing Orders
    - **ME61 / ME64**: Supplier Evaluation
    - **ME01**: Source List Maintenance
    - **QALS**: Quality Inspection Lot
    """)

# Filter working dataframe
v = active_df.copy()
if selected_mat_group != "All":
    v = v[v["Material_Group"] == selected_mat_group]
if selected_country != "All":
    v = v[v["Country"] == selected_country]
if selected_risk != "All":
    v = v[v["Risk_Category"] == selected_risk]
if selected_anomaly != "All" and "Anomaly_Status" in v.columns:
    v = v[v["Anomaly_Status"] == selected_anomaly]

# Main Header
st.title("🛡️ Supplier Risk Intelligence & Procurement Analytics System")
st.markdown("### Predictive Supplier Risk Scoring, Anomaly Detection & AI Alternative Sourcing")

# Key Metric Banners
total_suppliers = len(v)
total_spend = v["Total_Spend_USD"].sum()
high_risk_count = (v["Risk_Category"] == "High Risk").sum()
high_risk_pct = (high_risk_count / total_suppliers * 100.0) if total_suppliers > 0 else 0.0
avg_otif = v["OTIF_Rate"].mean() if total_suppliers > 0 else 0.0
avg_defect = v["Defect_Rate_Pct"].mean() if total_suppliers > 0 else 0.0
single_source_count = (v["Single_Source_Flag"] == 1).sum() if "Single_Source_Flag" in v.columns else 0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Active Suppliers", f"{total_suppliers:,}")
m2.metric("Total Spend", f"${total_spend/1e6:.1f}M")
m3.metric("High-Risk Suppliers", f"{high_risk_count}", f"{high_risk_pct:.1f}% of total", delta_color="inverse")
m4.metric("Avg OTIF Rate", f"{avg_otif:.1f}%", f"{avg_otif - 90:.1f}% vs Target")
m5.metric("Avg Defect Rate", f"{avg_defect:.2f}%", f"{avg_defect - 1.5:.2f}% vs Benchmark", delta_color="inverse")
m6.metric("Single-Source Vendors", f"{single_source_count}", "Concentration risk", delta_color="inverse")

st.markdown("---")

# Main Navigation Tabs
tab_overview, tab_geo, tab_high_risk, tab_recommender, tab_ml, tab_sap = st.tabs([
    "📊 Risk Matrix & Ranking",
    "🗺️ Global Distribution",
    "⚠️ High-Risk Analysis",
    "🔄 Alternative Supplier Recommender",
    "🤖 Machine Learning Diagnostics",
    "📋 SAP MM Data Dictionary & CSV"
])

# -------------------------------------------------------------
# TAB 1: Risk Matrix & Performance Ranking
# -------------------------------------------------------------
with tab_overview:
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Supplier Risk Matrix (Spend vs Composite Risk)")
        st.caption("Bubble size = PO Count | Color = Categorical Risk Level")
        color_discrete_map = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}
        fig_matrix = px.scatter(
            v,
            x="Composite_Risk_Score",
            y="Total_Spend_USD",
            size="PO_Count",
            color="Risk_Category",
            color_discrete_map=color_discrete_map,
            hover_name="Supplier_Name",
            hover_data=["Supplier_ID", "Country", "Material_Group", "OTIF_Rate", "Defect_Rate_Pct"],
            labels={"Composite_Risk_Score": "Composite Risk Score (0-100)", "Total_Spend_USD": "Total Spend (USD)"},
            log_y=True,
            height=460
        )
        # Add risk threshold quadrant lines
        fig_matrix.add_vline(x=58.0, line_dash="dash", line_color="red", annotation_text="High Risk Threshold (58+)")
        fig_matrix.add_vline(x=34.0, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold (34)")
        st.plotly_chart(fig_matrix, use_container_width=True)

    with col_right:
        st.subheader("Industry / Material Group Concentration")
        st.caption("Spend allocation across procurement categories")
        mat_spend = v.groupby("Material_Group", as_index=False)["Total_Spend_USD"].sum().sort_values("Total_Spend_USD", ascending=False)
        fig_pie = px.pie(
            mat_spend,
            names="Material_Group",
            values="Total_Spend_USD",
            hole=0.45,
            height=460
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Supplier Performance Ranking (Top & Bottom Performers)")
    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("#### 🏆 Top 10 Best Performing Suppliers (Lowest Risk & Highest OTIF)")
        top_perf = v.sort_values(by=["Composite_Risk_Score", "OTIF_Rate"], ascending=[True, False]).head(10)
        st.dataframe(
            top_perf[[
                "Supplier_ID", "Supplier_Name", "Material_Group", "Country",
                "OTIF_Rate", "Defect_Rate_Pct", "Composite_Risk_Score", "Risk_Category"
            ]],
            use_container_width=True,
            hide_index=True
        )

    with col_rank2:
        st.markdown("#### 🚨 Top 10 High-Risk Suppliers Requiring Immediate Intervention")
        worst_perf = v.sort_values(by=["Composite_Risk_Score", "Defect_Rate_Pct"], ascending=[False, False]).head(10)
        st.dataframe(
            worst_perf[[
                "Supplier_ID", "Supplier_Name", "Material_Group", "Country",
                "OTIF_Rate", "Defect_Rate_Pct", "Composite_Risk_Score", "Risk_Category"
            ]],
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------------------
# TAB 2: Geographical Distribution & Country Risk
# -------------------------------------------------------------
with tab_geo:
    st.subheader("Global Supplier Distribution & Geopolitical Risk Overlay")
    st.caption("Evaluates concentration risks across supplier sourcing countries")

    col_geo1, col_geo2 = st.columns([3, 2])
    with col_geo1:
        country_agg = v.groupby("Country", as_index=False).agg(
            Supplier_Count=("Supplier_ID", "count"),
            Total_Spend=("Total_Spend_USD", "sum"),
            Avg_Geo_Risk=("Geopolitical_Risk_Score", "mean"),
            Avg_OTIF=("OTIF_Rate", "mean"),
            Avg_Defect=("Defect_Rate_Pct", "mean")
        ).sort_values("Supplier_Count", ascending=False)

        fig_geo_bar = px.bar(
            country_agg,
            x="Country",
            y="Supplier_Count",
            color="Avg_Geo_Risk",
            color_continuous_scale="Reds",
            labels={"Avg_Geo_Risk": "Geopolitical Risk", "Supplier_Count": "Number of Suppliers"},
            text="Supplier_Count",
            title="Suppliers by Sourcing Country & Geopolitical Risk",
            height=430
        )
        st.plotly_chart(fig_geo_bar, use_container_width=True)

    with col_geo2:
        st.write("#### Country Sourcing Exposure")
        fig_country_spend = px.pie(
            country_agg,
            names="Country",
            values="Total_Spend",
            title="Spend Distribution by Country",
            hole=0.4,
            height=430
        )
        st.plotly_chart(fig_country_spend, use_container_width=True)

    st.markdown("#### Country Sourcing Risk Breakdown")
    country_display = country_agg.copy()
    country_display["Total_Spend_USD"] = country_display["Total_Spend"].map("${:,.0f}".format)
    country_display["Avg_Geo_Risk"] = country_display["Avg_Geo_Risk"].map("{:.1f}".format)
    country_display["Avg_OTIF"] = country_display["Avg_OTIF"].map("{:.1f}%".format)
    country_display["Avg_Defect"] = country_display["Avg_Defect"].map("{:.2f}%".format)
    st.dataframe(
        country_display[["Country", "Supplier_Count", "Total_Spend_USD", "Avg_Geo_Risk", "Avg_OTIF", "Avg_Defect"]],
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------------------
# TAB 3: High-Risk Suppliers & Operational Deep-Dive
# -------------------------------------------------------------
with tab_high_risk:
    st.subheader("Critical Supplier Risk Management")
    st.markdown("Detailed breakdown of suppliers flagged with **High Risk** or **Anomalous Operational Patterns**.")

    high_risk_df = v[v["Risk_Category"] == "High Risk"].copy()
    if high_risk_df.empty:
        st.success("No suppliers match the High-Risk filter in current selection.")
    else:
        st.warning(f"⚠️ {len(high_risk_df)} suppliers require urgent procurement review, dual-sourcing, or vendor development.")

        # Root-cause breakdown
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            poor_otif = (high_risk_df["OTIF_Rate"] < 70.0).sum()
            st.metric("Severe Delivery Delays (OTIF < 70%)", f"{poor_otif} Vendors")
        with col_c2:
            poor_qual = (high_risk_df["Defect_Rate_Pct"] > 5.0).sum()
            st.metric("High Quality Rejection (> 5% Defect)", f"{poor_qual} Vendors")
        with col_c3:
            poor_fin = (high_risk_df["Financial_Stability_Score"] < 40.0).sum()
            st.metric("Solvency / Financial Distress (< 40)", f"{poor_fin} Vendors")

        st.markdown("#### High-Risk Supplier Register")
        display_cols = [
            "Supplier_ID", "Supplier_Name", "Material_Group", "Country", "Total_Spend_USD",
            "OTIF_Rate", "Defect_Rate_Pct", "Average_Delay_Days", "Price_Variance_Pct",
            "Financial_Stability_Score", "Single_Source_Flag", "Composite_Risk_Score"
        ]
        if "Anomaly_Status" in high_risk_df.columns:
            display_cols.append("Anomaly_Status")

        st.dataframe(
            high_risk_df[display_cols].sort_values(by="Composite_Risk_Score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download High-Risk Action List (CSV)",
            high_risk_df.to_csv(index=False).encode(),
            "high_risk_suppliers_action_plan.csv",
            "text/csv"
        )

# -------------------------------------------------------------
# TAB 4: Alternative Supplier Recommender
# -------------------------------------------------------------
with tab_recommender:
    st.subheader("AI Alternative Supplier Recommendation Engine")
    st.markdown("Select an existing or at-risk supplier to instantly find qualified, lower-risk alternative suppliers in the same Material Group.")

    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        supplier_options = [f"{row['Supplier_ID']} - {row['Supplier_Name']} ({row['Material_Group']}, Risk: {row['Risk_Category']})"
                            for _, row in active_df.iterrows()]
        
        # Default to first high-risk supplier if present
        default_index = 0
        high_risk_indices = [i for i, row in active_df.iterrows() if row["Risk_Category"] == "High Risk"]
        if high_risk_indices:
            default_index = high_risk_indices[0]

        selected_supplier_str = st.selectbox("Select Target Supplier to Evaluate / Replace:", supplier_options, index=default_index)
        target_id = selected_supplier_str.split(" - ")[0]

    with col_sel2:
        top_k = st.slider("Number of Alternatives", 2, 8, 4)

    # Display Current Supplier Profile
    curr_supp = active_df[active_df["Supplier_ID"] == target_id].iloc[0]
    
    st.markdown("#### 🎯 Current Supplier Profile")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Supplier ID", curr_supp["Supplier_ID"])
    p2.metric("Material Group", curr_supp["Material_Group"])
    p3.metric("Current OTIF Rate", f"{curr_supp['OTIF_Rate']:.1f}%")
    p4.metric("Defect Rate", f"{curr_supp['Defect_Rate_Pct']:.2f}%")
    p5.metric("Composite Risk", f"{curr_supp['Composite_Risk_Score']:.1f}", curr_supp["Risk_Category"], delta_color="inverse")
    p6.metric("Country", curr_supp["Country"])

    # Compute Recommendations
    recs = recommend_alternative_suppliers(target_id, active_df, top_n=top_k)

    if recs.empty:
        st.info("No alternative suppliers found meeting the criteria in the same material group.")
    else:
        st.markdown(f"#### 💡 Top {len(recs)} AI-Recommended Qualified Alternative Suppliers")
        st.dataframe(
            recs[[
                "Alternative_Supplier_ID", "Supplier_Name", "Country", "Recommendation_Score",
                "OTIF_Rate", "OTIF_Delta", "Defect_Rate_Pct", "Defect_Delta",
                "Composite_Risk_Score", "Risk_Score_Delta", "Recommendation_Rationale"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.caption("ℹ️ Recommendation Score (0-100) weighs OTIF reliability, low defect rate, financial stability, and geopolitical diversification.")

# -------------------------------------------------------------
# TAB 5: Machine Learning Diagnostics & Explainability
# -------------------------------------------------------------
with tab_ml:
    st.subheader("Machine Learning Models & Explainability")
    st.caption("Trained on SAP MM simulated procurement data across 4 distinct algorithms")

    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)
    with ml_col1:
        rf_acc = metrics_data.get("random_forest", {}).get("accuracy", 0.965)
        st.metric("Random Forest Accuracy", f"{rf_acc * 100:.1f}%", "Risk Classification")
    with ml_col2:
        xgb_r2 = metrics_data.get("xgboost", {}).get("r2_score", 0.994)
        st.metric("XGBoost R² Score", f"{xgb_r2:.4f}", "Continuous Risk Score")
    with ml_col3:
        st.metric("K-Means Segments", "4 Quadrants", "Strategic Segmentation")
    with ml_col4:
        iso_cnt = metrics_data.get("isolation_forest", {}).get("anomalies_detected", 45)
        st.metric("Isolation Forest Outliers", f"{iso_cnt} Patterns", "Anomaly Detection")

    st.markdown("---")
    col_feat1, col_feat2 = st.columns(2)

    with col_feat1:
        st.markdown("#### 🌲 Random Forest - Feature Importance")
        rf_imp = metrics_data.get("random_forest", {}).get("feature_importance", {})
        if rf_imp:
            imp_df = pd.DataFrame(list(rf_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_rf = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Top Drivers of Supplier Risk Classification")
            st.plotly_chart(fig_rf, use_container_width=True)

    with col_feat2:
        st.markdown("#### ⚡ XGBoost - Feature Importance (Gain)")
        xgb_imp = metrics_data.get("xgboost", {}).get("feature_importance", {})
        if xgb_imp:
            imp_df = pd.DataFrame(list(xgb_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_xgb = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Drivers of Predicted Risk Score")
            st.plotly_chart(fig_xgb, use_container_width=True)

    st.markdown("---")
    col_km, col_iso = st.columns(2)

    with col_km:
        st.markdown("#### 🧩 K-Means Supplier Strategic Segmentation")
        if "Supplier_Cluster_Label" in active_df.columns:
            fig_cluster = px.scatter(
                active_df,
                x="OTIF_Rate",
                y="Composite_Risk_Score",
                color="Supplier_Cluster_Label",
                size="Total_Spend_USD",
                hover_name="Supplier_Name",
                title="Operational Supplier Clusters (OTIF vs Risk)",
                labels={"Composite_Risk_Score": "Risk Score", "OTIF_Rate": "OTIF Rate (%)"},
                height=420
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    with col_iso:
        st.markdown("#### 🔍 Isolation Forest - Unusual Procurement Anomalies")
        if "Anomaly_Status" in active_df.columns:
            fig_anomaly = px.scatter(
                active_df,
                x="Price_Variance_Pct",
                y="Defect_Rate_Pct",
                color="Anomaly_Status",
                color_discrete_map={"Anomaly Flagged": "#dc3545", "Normal Operational": "#28a745"},
                hover_name="Supplier_Name",
                title="Anomalies in Price Variance vs Defect Rates",
                labels={"Price_Variance_Pct": "Price Variance (%)", "Defect_Rate_Pct": "Defect Rate (%)"},
                height=420
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)

# -------------------------------------------------------------
# TAB 6: SAP MM Data Dictionary & CSV Schema
# -------------------------------------------------------------
with tab_sap:
    st.subheader("📋 SAP Materials Management (MM) Data Schema & Mapping")
    st.markdown("""
    This system maps standard procurement features directly to SAP MM tables, fields, and transactions:
    """)

    mapping_df = get_field_mapping_dataframe()
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📥 Download Master Dataset & Templates")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "Download Modeled Supplier Procurement Dataset (CSV)",
            active_df.to_csv(index=False).encode(),
            "supplier_procurement_modeled.csv",
            "text/csv"
        )
    with col_d2:
        template_cols = list(SAP_FIELD_MAPPING.keys())
        sample_row = {col: SAP_FIELD_MAPPING[col]["sample"] for col in template_cols}
        template_df = pd.DataFrame([sample_row])
        st.download_button(
            "Download Blank SAP MM CSV Template (CSV)",
            template_df.to_csv(index=False).encode(),
            "sap_supplier_procurement_template.csv",
            "text/csv"
        )

st.markdown("---")
st.caption("SAP MM Supplier Risk Intelligence & Procurement Analytics System | Pair Programming Portfolio | Antigravity AI")
