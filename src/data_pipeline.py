"""
Data Ingestion, Validation, Cleaning & Feature Engineering Pipeline for Supplier Risk.
Ensures compatibility with both default datasets and user-uploaded custom CSV files.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env_setup  # Windows Anaconda DLL bootstrap
from src.sap_mapping import SAP_FIELD_MAPPING, REQUIRED_COLUMNS
DATA_DIR = ROOT / "data"

FEATURE_COLUMNS = [
    "OTIF_Rate",
    "Defect_Rate_Pct",
    "Average_Delay_Days",
    "Lead_Time_Days",
    "Price_Variance_Pct",
    "Financial_Stability_Score",
    "Geopolitical_Risk_Score",
    "Single_Source_Flag",
    "Contract_Compliance_Rate",
    "ESG_Score",
    "Total_Spend_USD",
    "PO_Count"
]

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate incoming dataframe against SAP MM schema, handle missing columns,
    compute derived metrics, and clean data.
    """
    cleaned = df.copy()

    # Column name normalization (strip whitespace, standardize case)
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # Check for minimal identifier columns
    if "Supplier_ID" not in cleaned.columns:
        cleaned["Supplier_ID"] = [f"VEND-{10000 + i}" for i in range(1, len(cleaned) + 1)]
    if "Supplier_Name" not in cleaned.columns:
        cleaned["Supplier_Name"] = [f"Vendor {sid}" for sid in cleaned["Supplier_ID"]]
    if "Material_Group" not in cleaned.columns:
        cleaned["Material_Group"] = "General Procurement"
    if "Country" not in cleaned.columns:
        cleaned["Country"] = "Global"
    if "City" not in cleaned.columns:
        cleaned["City"] = "Unknown"

    # Compute OTIF if missing but components exist
    if "OTIF_Rate" not in cleaned.columns:
        if "On_Time_Delivery_Rate" in cleaned.columns and "In_Full_Delivery_Rate" in cleaned.columns:
            cleaned["OTIF_Rate"] = (cleaned["On_Time_Delivery_Rate"] * cleaned["In_Full_Delivery_Rate"]) / 100.0
        elif "On_Time_Delivery_Rate" in cleaned.columns:
            cleaned["OTIF_Rate"] = cleaned["On_Time_Delivery_Rate"]
        else:
            cleaned["OTIF_Rate"] = 85.0

    # Ensure required numerical features exist with domain sensible defaults
    numeric_defaults = {
        "Defect_Rate_Pct": 2.0,
        "Average_Delay_Days": 3.0,
        "Lead_Time_Days": 30.0,
        "Price_Variance_Pct": 1.5,
        "Financial_Stability_Score": 70.0,
        "Geopolitical_Risk_Score": 25.0,
        "Single_Source_Flag": 0,
        "Contract_Compliance_Rate": 85.0,
        "ESG_Score": 75.0,
        "Total_Spend_USD": 500000.0,
        "PO_Count": 50
    }

    for col, default_val in numeric_defaults.items():
        if col not in cleaned.columns:
            cleaned[col] = default_val
        else:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(default_val)

    # Derived Feature 1: Spend share in material group (Supplier concentration / dependency)
    mat_group_spend = cleaned.groupby("Material_Group")["Total_Spend_USD"].transform("sum")
    cleaned["Spend_Share_In_Group"] = (cleaned["Total_Spend_USD"] / mat_group_spend.replace(0, 1) * 100.0).round(2)

    # Derived Feature 2: High concentration risk (Single source or spend share > 30%)
    cleaned["High_Concentration_Risk"] = (
        (cleaned["Single_Source_Flag"] == 1) | (cleaned["Spend_Share_In_Group"] > 30.0)
    ).astype(int)

    # Calculate Composite Risk Score if missing
    if "Composite_Risk_Score" not in cleaned.columns:
        delivery_score = np.clip((100.0 - cleaned["OTIF_Rate"]) * 0.55, 0.0, 30.0)
        quality_score = np.clip((cleaned["Defect_Rate_Pct"] / 12.0) * 25.0, 0.0, 30.0)
        solvency_score = np.clip(((100.0 - cleaned["Financial_Stability_Score"]) / 100.0) * 20.0, 0.0, 20.0)
        price_score = np.clip((np.maximum(0.0, cleaned["Price_Variance_Pct"]) / 20.0) * 15.0, 0.0, 15.0)
        geo_score = np.clip((cleaned["Geopolitical_Risk_Score"] / 100.0) * 10.0, 0.0, 10.0)
        single_source_score = cleaned["Single_Source_Flag"] * 10.0

        raw_risk = delivery_score + quality_score + solvency_score + price_score + geo_score + single_source_score
        cleaned["Composite_Risk_Score"] = np.clip(raw_risk, 5.0, 98.0).round(1)

    # Assign categorical Risk_Category if missing
    if "Risk_Category" not in cleaned.columns:
        conditions = [
            cleaned["Composite_Risk_Score"] >= 58.0,
            cleaned["Composite_Risk_Score"] >= 34.0
        ]
        choices = ["High Risk", "Medium Risk"]
        cleaned["Risk_Category"] = np.select(conditions, choices, default="Low Risk")

    return cleaned

def prepare_pipeline_data(csv_path: Path = None) -> pd.DataFrame:
    """Load, validate, feature-engineer, and save processed supplier dataset."""
    if csv_path is None:
        csv_path = DATA_DIR / "supplier_procurement_data.csv"

    if not csv_path.exists():
        from src.generate_supplier_data import generate_dataset
        print("Supplier dataset not found. Generating default simulated SAP MM procurement dataset...")
        generate_dataset()

    df = pd.read_csv(csv_path)
    processed_df = validate_and_clean_data(df)
    
    out_path = DATA_DIR / "supplier_master_processed.csv"
    processed_df.to_csv(out_path, index=False)
    print(f"Data pipeline complete. Processed {len(processed_df)} records saved to {out_path}")
    return processed_df

if __name__ == "__main__":
    prepare_pipeline_data()
