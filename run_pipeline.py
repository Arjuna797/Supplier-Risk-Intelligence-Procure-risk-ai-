"""
Supplier Risk Intelligence & Procurement Analytics System - Master Pipeline Runner
Executes:
1. Data generation / validation (SAP MM schema alignment)
2. Cleaning & Feature Engineering
3. Machine Learning Training (Random Forest, XGBoost, KMeans, Isolation Forest)
4. Model evaluation & artifact persistence
"""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env_setup  # Windows DLL bootstrap
from src.generate_supplier_data import generate_dataset
from src.data_pipeline import prepare_pipeline_data
from src.train_models import train_all_models

def main():
    parser = argparse.ArgumentParser(description="Supplier Risk Intelligence Pipeline Runner")
    parser.add_argument("--regenerate-data", action="store_true", help="Force regenerate simulated supplier dataset")
    parser.add_argument("--csv", type=str, default=None, help="Custom CSV dataset path")
    args = parser.parse_args()

    print("=" * 75)
    print(" SAP MM SUPPLIER RISK INTELLIGENCE & PROCUREMENT ANALYTICS SYSTEM")
    print("=" * 75)

    csv_path = Path(args.csv) if args.csv else None
    data_file = ROOT / "data" / "supplier_procurement_data.csv"

    if args.regenerate_data or (csv_path is None and not data_file.exists()):
        print("\n[Step 1/3] Generating realistic SAP MM procurement dataset...")
        generate_dataset(n_samples=1000)
    else:
        print(f"\n[Step 1/3] Using dataset: {csv_path or data_file}")

    print("\n[Step 2/3] Running Data Ingestion & Feature Engineering Pipeline...")
    prepare_pipeline_data(csv_path)

    print("\n[Step 3/3] Training Machine Learning Models...")
    metrics = train_all_models()

    print("\n" + "=" * 75)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Random Forest Accuracy   : {metrics['random_forest']['accuracy'] * 100:.2f}% (Macro F1: {metrics['random_forest']['macro_f1']:.4f})")
    print(f" XGBoost Risk Score R2    : {metrics['xgboost']['r2_score']:.4f} (RMSE: {metrics['xgboost']['rmse']:.2f})")
    print(f" K-Means Clusters Built   : {metrics['kmeans']['n_clusters']} Operational Segments")
    print(f" Isolation Forest Anomalies: {metrics['isolation_forest']['anomalies_detected']} Suppliers Flagged")
    print("=" * 75)
    print("\nTo launch the interactive dashboard, run:")
    print("    streamlit run app.py")
    print("=" * 75)

if __name__ == "__main__":
    main()
