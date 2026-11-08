"""
Alternative Supplier Recommendation Engine.
Matches high-risk or bottleneck vendors with top-performing qualified alternative
suppliers within the same Material Group (SAP MATKL) to mitigate supply disruption.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env_setup

def recommend_alternative_suppliers(
    target_supplier_id: str,
    df: pd.DataFrame,
    top_n: int = 5,
    allow_cross_country: bool = True
) -> pd.DataFrame:
    """
    Recommend the best alternative suppliers for a given supplier ID.
    Criteria:
    - Same Material_Group (SAP MATKL)
    - Significantly lower risk score than target
    - High OTIF (>= 85%) and low defect rate (<= 3.5%)
    - Capacity & stability compatibility
    """
    target = df[df["Supplier_ID"] == target_supplier_id]
    if target.empty:
        return pd.DataFrame()

    target_row = target.iloc[0]
    mat_group = target_row["Material_Group"]
    target_otif = target_row["OTIF_Rate"]
    target_defect = target_row["Defect_Rate_Pct"]
    target_risk = target_row["Composite_Risk_Score"]
    target_country = target_row["Country"]

    # Filter candidate pool
    candidates = df[
        (df["Material_Group"] == mat_group) &
        (df["Supplier_ID"] != target_supplier_id) &
        (df["Composite_Risk_Score"] < max(40.0, target_risk - 10.0))
    ].copy()

    if candidates.empty:
        # Fallback: Relax risk threshold if pool is small
        candidates = df[
            (df["Material_Group"] == mat_group) &
            (df["Supplier_ID"] != target_supplier_id)
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Calculate multi-dimensional recommendation suitability score (0 - 100)
    # Higher is better
    recommendations = []
    for _, cand in candidates.iterrows():
        # 1. Delivery reliability score (40% weight)
        otif_score = cand["OTIF_Rate"]
        # 2. Quality performance score (30% weight)
        quality_score = max(0.0, 100.0 - (cand["Defect_Rate_Pct"] * 10.0))
        # 3. Solvency & Financial health (15% weight)
        solvency_score = cand["Financial_Stability_Score"]
        # 4. Geopolitical / Country resilience bonus (15% weight)
        geo_bonus = max(0.0, 100.0 - cand["Geopolitical_Risk_Score"])
        if allow_cross_country and cand["Country"] != target_country:
            # Diversification bonus against regional shocks
            geo_bonus = min(100.0, geo_bonus * 1.1)

        match_score = (
            otif_score * 0.40 +
            quality_score * 0.30 +
            solvency_score * 0.15 +
            geo_bonus * 0.15
        )

        otif_gain = round(cand["OTIF_Rate"] - target_otif, 2)
        defect_reduction = round(target_defect - cand["Defect_Rate_Pct"], 2)
        risk_reduction = round(target_risk - cand["Composite_Risk_Score"], 2)

        rationale_parts = []
        if otif_gain > 0:
            rationale_parts.append(f"+{otif_gain}% OTIF gain")
        if defect_reduction > 0:
            rationale_parts.append(f"-{defect_reduction}% defect reduction")
        if risk_reduction > 0:
            rationale_parts.append(f"-{risk_reduction} risk score drop")
        if cand["Country"] != target_country:
            rationale_parts.append(f"Geo-diversified ({cand['Country']})")

        rationale = " | ".join(rationale_parts) if rationale_parts else "Meets operational criteria"

        recommendations.append({
            "Alternative_Supplier_ID": cand["Supplier_ID"],
            "Supplier_Name": cand["Supplier_Name"],
            "Country": cand["Country"],
            "City": cand["City"],
            "Material_Group": cand["Material_Group"],
            "Recommendation_Score": round(float(match_score), 1),
            "OTIF_Rate": cand["OTIF_Rate"],
            "OTIF_Delta": otif_gain,
            "Defect_Rate_Pct": cand["Defect_Rate_Pct"],
            "Defect_Delta": defect_reduction,
            "Composite_Risk_Score": cand["Composite_Risk_Score"],
            "Risk_Score_Delta": risk_reduction,
            "Financial_Stability_Score": cand["Financial_Stability_Score"],
            "Total_Spend_USD": cand["Total_Spend_USD"],
            "Recommendation_Rationale": rationale
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values(by="Recommendation_Score", ascending=False).head(top_n)
    return rec_df
