"""
Generates realistic, production-grade SAP MM simulated supplier procurement dataset.
Includes delivery performance, defect rates, lead times, price variances,
financial health, geopolitical risk, and order history.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

COUNTRIES_DATA = [
    {"Country": "Germany", "City": "Stuttgart", "Geo_Risk": 15.0},
    {"Country": "Germany", "City": "Munich", "Geo_Risk": 15.0},
    {"Country": "USA", "City": "Chicago", "Geo_Risk": 18.0},
    {"Country": "USA", "City": "Detroit", "Geo_Risk": 18.0},
    {"Country": "USA", "City": "San Jose", "Geo_Risk": 18.0},
    {"Country": "Japan", "City": "Tokyo", "Geo_Risk": 16.0},
    {"Country": "South Korea", "City": "Seoul", "Geo_Risk": 24.0},
    {"Country": "Taiwan", "City": "Hsinchu", "Geo_Risk": 44.0},
    {"Country": "China", "City": "Shenzhen", "Geo_Risk": 52.0},
    {"Country": "China", "City": "Shanghai", "Geo_Risk": 48.0},
    {"Country": "India", "City": "Bangalore", "Geo_Risk": 38.0},
    {"Country": "India", "City": "Pune", "Geo_Risk": 38.0},
    {"Country": "Mexico", "City": "Monterrey", "Geo_Risk": 42.0},
    {"Country": "Vietnam", "City": "Ho Chi Minh City", "Geo_Risk": 42.0},
    {"Country": "UK", "City": "Birmingham", "Geo_Risk": 20.0},
    {"Country": "France", "City": "Lyon", "Geo_Risk": 19.0},
    {"Country": "Italy", "City": "Turin", "Geo_Risk": 23.0},
    {"Country": "Poland", "City": "Wroclaw", "Geo_Risk": 26.0},
]

MATERIAL_GROUPS = [
    "Electronics & Sensors",
    "Mechanical & Machined Parts",
    "Raw Materials & Steel",
    "Industrial Chemicals & Polymers",
    "Packaging & Containers",
    "Automotive Assemblies",
    "Electrical Equipment",
    "IT Hardware & Telecom Equipment"
]

SUPPLIER_PREFIXES = [
    "Apex", "Precision", "Global", "Nova", "Titan", "Vanguard", "Summit", "Nexus",
    "Optima", "Zenith", "Quantum", "Alliance", "Pinnacle", "Dynamic", "Stellar",
    "Alpha", "Omega", "Matrix", "Aero", "Euro", "Pacific", "Continental", "Integrated"
]

SUPPLIER_SUFFIXES = [
    "Components", "Technologies", "Manufacturing", "Industries", "Solutions",
    "Systems", "Engineering", "Materials", "Logistics", "Corp", "GmbH", "S.A."
]

PURCHASING_ORGS = ["1000", "2000", "3000", "4000"]
PAYMENT_TERMS = ["NT30", "NT60", "NT90", "2/10 Net 30"]

def generate_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate simulated supplier procurement dataset aligned with SAP MM evaluation."""
    np.random.seed(random_state)
    records = []

    for i in range(1, n_samples + 1):
        supplier_id = f"VEND-{10000 + i}"
        pref = np.random.choice(SUPPLIER_PREFIXES)
        suff = np.random.choice(SUPPLIER_SUFFIXES)
        supplier_name = f"{pref} {suff} #{i}"
        
        geo_info = np.random.choice(COUNTRIES_DATA)
        country = geo_info["Country"]
        city = geo_info["City"]
        geo_risk = geo_info["Geo_Risk"] + np.random.uniform(-4.0, 4.0)
        geo_risk = np.clip(round(geo_risk, 1), 8.0, 95.0)

        mat_group = np.random.choice(MATERIAL_GROUPS)
        p_org = np.random.choice(PURCHASING_ORGS)
        p_terms = np.random.choice(PAYMENT_TERMS)

        # Baseline supplier profile distribution (55% good, 30% medium, 15% high-risk)
        profile_type = np.random.choice(["good", "medium", "poor"], p=[0.55, 0.30, 0.15])
        
        if profile_type == "good":
            on_time = np.random.normal(95.0, 3.0)
            in_full = np.random.normal(96.5, 2.5)
            defect_rate = np.random.exponential(0.9)
            avg_delay = np.random.exponential(1.5)
            lead_time = np.random.normal(25.0, 6.0)
            price_var = np.random.normal(0.5, 1.5)
            fin_stability = np.random.normal(84.0, 7.0)
            contract_comp = np.random.normal(95.0, 3.5)
            esg_score = np.random.normal(86.0, 6.5)
            single_source_prob = 0.08
        elif profile_type == "medium":
            on_time = np.random.normal(82.0, 5.0)
            in_full = np.random.normal(86.0, 4.0)
            defect_rate = np.random.exponential(2.8) + 1.2
            avg_delay = np.random.exponential(4.2) + 2.0
            lead_time = np.random.normal(48.0, 11.0)
            price_var = np.random.normal(4.8, 2.8)
            fin_stability = np.random.normal(58.0, 9.0)
            contract_comp = np.random.normal(80.0, 7.0)
            esg_score = np.random.normal(66.0, 8.5)
            single_source_prob = 0.22
        else:  # poor profile
            on_time = np.random.normal(64.0, 7.5)
            in_full = np.random.normal(68.0, 7.0)
            defect_rate = np.random.exponential(5.5) + 3.8
            avg_delay = np.random.exponential(9.0) + 6.0
            lead_time = np.random.normal(75.0, 18.0)
            price_var = np.random.normal(11.0, 4.5)
            fin_stability = np.random.normal(34.0, 10.0)
            contract_comp = np.random.normal(60.0, 10.0)
            esg_score = np.random.normal(42.0, 9.5)
            single_source_prob = 0.45

        # Boundary clipping
        on_time = np.clip(round(on_time, 2), 35.0, 99.8)
        in_full = np.clip(round(in_full, 2), 40.0, 100.0)
        otif_rate = round((on_time * in_full) / 100.0, 2)
        defect_rate = np.clip(round(defect_rate, 2), 0.05, 20.0)
        avg_delay = np.clip(round(avg_delay, 1), 0.0, 50.0)
        lead_time = np.clip(round(lead_time, 0), 5.0, 160.0)
        price_var = np.clip(round(price_var, 2), -5.0, 35.0)
        fin_stability = np.clip(round(fin_stability, 1), 5.0, 99.0)
        contract_comp = np.clip(round(contract_comp, 1), 35.0, 100.0)
        esg_score = np.clip(round(esg_score, 1), 15.0, 99.0)

        single_source = int(np.random.rand() < single_source_prob)
        po_count = int(np.random.randint(15, 450))
        base_spend = np.random.lognormal(mean=13.2, sigma=1.0)
        spend_usd = round(float(np.clip(base_spend, 50000, 40000000)), 2)

        # Inject 4% targeted operational anomalies for Isolation Forest discovery
        if i % 25 == 0:
            anomaly_type = np.random.choice(["price_spike", "severe_delay", "quality_collapse", "financial_distress"])
            if anomaly_type == "price_spike":
                price_var = round(price_var + np.random.uniform(20.0, 35.0), 2)
            elif anomaly_type == "severe_delay":
                avg_delay = round(avg_delay + np.random.uniform(25.0, 45.0), 1)
                on_time = round(max(30.0, on_time - 35.0), 2)
                otif_rate = round((on_time * in_full) / 100.0, 2)
            elif anomaly_type == "quality_collapse":
                defect_rate = round(defect_rate + np.random.uniform(10.0, 18.0), 2)
            elif anomaly_type == "financial_distress":
                fin_stability = round(max(5.0, fin_stability - 40.0), 1)
                spend_usd = round(spend_usd * 2.5, 2)

        # Composite Risk Formula (0 to 100) aligned with SAP MM Supplier Evaluation (ME61):
        # 1. Delivery failure component (0-25 pts): OTIF underperformance
        delivery_score = np.clip((100.0 - otif_rate) * 0.55, 0.0, 30.0)
        # 2. Quality defect component (0-25 pts)
        quality_score = np.clip((defect_rate / 12.0) * 25.0, 0.0, 30.0)
        # 3. Solvency & Financial risk (0-20 pts)
        solvency_score = np.clip(((100.0 - fin_stability) / 100.0) * 20.0, 0.0, 20.0)
        # 4. Price & Cost variance risk (0-15 pts)
        price_score = np.clip((max(0.0, price_var) / 20.0) * 15.0, 0.0, 15.0)
        # 5. Geopolitical risk (0-10 pts)
        geo_score = np.clip((geo_risk / 100.0) * 10.0, 0.0, 10.0)
        # 6. Single-source exposure (0-10 pts)
        single_source_score = single_source * 10.0

        raw_risk = delivery_score + quality_score + solvency_score + price_score + geo_score + single_source_score
        composite_risk = round(float(np.clip(raw_risk, 5.0, 98.0)), 1)

        # Categorical Risk Classification:
        # High Risk: >= 58.0
        # Medium Risk: 34.0 to 57.9
        # Low Risk: < 34.0
        if composite_risk >= 58.0:
            risk_category = "High Risk"
        elif composite_risk >= 34.0:
            risk_category = "Medium Risk"
        else:
            risk_category = "Low Risk"

        records.append({
            "Supplier_ID": supplier_id,
            "Supplier_Name": supplier_name,
            "Country": country,
            "City": city,
            "Material_Group": mat_group,
            "Purchasing_Org": p_org,
            "Payment_Terms": p_terms,
            "PO_Count": po_count,
            "Total_Spend_USD": spend_usd,
            "On_Time_Delivery_Rate": on_time,
            "In_Full_Delivery_Rate": in_full,
            "OTIF_Rate": otif_rate,
            "Defect_Rate_Pct": defect_rate,
            "Average_Delay_Days": avg_delay,
            "Lead_Time_Days": int(lead_time),
            "Price_Variance_Pct": price_var,
            "Financial_Stability_Score": fin_stability,
            "Geopolitical_Risk_Score": geo_risk,
            "Single_Source_Flag": single_source,
            "Contract_Compliance_Rate": contract_comp,
            "ESG_Score": esg_score,
            "Composite_Risk_Score": composite_risk,
            "Risk_Category": risk_category
        })

    df = pd.DataFrame(records)
    out_file = DATA_DIR / "supplier_procurement_data.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} supplier procurement records at: {out_file}")
    print(f"Risk Category breakdown:\n{df['Risk_Category'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_dataset()
