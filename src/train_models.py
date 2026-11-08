"""
Machine Learning Training Pipeline for Supplier Risk Intelligence.
Trains:
1. Random Forest Classifier: Multi-class supplier risk categorization (Low / Medium / High).
2. XGBoost Regressor: Continuous supplier risk score prediction (0 - 100).
3. K-Means Clustering: Operational supplier segmentation into strategic procurement quadrants.
4. Isolation Forest: Unsupervised anomaly detection for unusual procurement patterns.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env_setup  # Windows Anaconda DLL bootstrap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.cluster import KMeans
import xgboost as xgb

from src.data_pipeline import FEATURE_COLUMNS, prepare_pipeline_data

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_all_models(data_path: Path = None):
    """Train all 4 models, evaluate performance, save models and export modeled dataset."""
    if data_path is None or not data_path.exists():
        data_path = DATA_DIR / "supplier_master_processed.csv"
        if not data_path.exists():
            prepare_pipeline_data()

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records for model training.")

    X = df[FEATURE_COLUMNS].copy()
    y_cat = df["Risk_Category"].copy()
    y_num = df["Composite_Risk_Score"].copy()

    # Train / Test split (80 / 20)
    X_train, X_test, y_cat_train, y_cat_test, y_num_train, y_num_test = train_test_split(
        X, y_cat, y_num, test_size=0.20, random_state=42, stratify=y_cat
    )

    # Standard Scaler for distance-based & neural models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    # -------------------------------------------------------------
    # 1. Random Forest Classifier (Risk Classification)
    # -------------------------------------------------------------
    print("\n--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_cat_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    rf_acc = accuracy_score(y_cat_test, rf_preds)
    rf_f1_macro = f1_score(y_cat_test, rf_preds, average="macro")
    rf_report = classification_report(y_cat_test, rf_preds, output_dict=True)
    rf_cm = confusion_matrix(y_cat_test, rf_preds, labels=["Low Risk", "Medium Risk", "High Risk"]).tolist()

    rf_importance = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_model.feature_importances_]))
    rf_importance_sorted = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))

    print(f"Random Forest Accuracy: {rf_acc:.4f} | Macro F1: {rf_f1_macro:.4f}")

    # -------------------------------------------------------------
    # 2. XGBoost Regressor (Continuous Risk Score Prediction)
    # -------------------------------------------------------------
    print("\n--- Training XGBoost Risk Predictor ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_num_train)
    xgb_preds = xgb_model.predict(X_test)

    xgb_rmse = float(np.sqrt(mean_squared_error(y_num_test, xgb_preds)))
    xgb_mae = float(mean_absolute_error(y_num_test, xgb_preds))
    xgb_r2 = float(r2_score(y_num_test, xgb_preds))

    xgb_importance = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in xgb_model.feature_importances_]))
    xgb_importance_sorted = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True))

    print(f"XGBoost R²: {xgb_r2:.4f} | RMSE: {xgb_rmse:.4f} | MAE: {xgb_mae:.4f}")

    # -------------------------------------------------------------
    # 3. K-Means Supplier Segmentation
    # -------------------------------------------------------------
    print("\n--- Training K-Means Supplier Segmentation ---")
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=20, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(X_all_scaled)

    # Dynamic labeling based on cluster centers
    cluster_df = df.copy()
    cluster_df["cluster"] = cluster_labels
    cluster_profiles = cluster_df.groupby("cluster")[["Total_Spend_USD", "OTIF_Rate", "Defect_Rate_Pct", "Composite_Risk_Score"]].mean()

    # Map clusters to meaningful SAP procurement quadrant names
    cluster_name_map = {}
    for c_id, row in cluster_profiles.iterrows():
        spend = row["Total_Spend_USD"]
        otif = row["OTIF_Rate"]
        risk = row["Composite_Risk_Score"]
        defects = row["Defect_Rate_Pct"]

        if risk >= 55.0 or defects >= 4.5:
            cluster_name_map[c_id] = "Critical Risk & Bottleneck"
        elif spend >= cluster_profiles["Total_Spend_USD"].median() and otif >= 90.0:
            cluster_name_map[c_id] = "Strategic Key Partners"
        elif defects > cluster_profiles["Defect_Rate_Pct"].median() and otif < 85.0:
            cluster_name_map[c_id] = "Underperforming / Action Plan"
        else:
            cluster_name_map[c_id] = "Operational Core Performers"

    # Ensure uniqueness of names if collisions occur
    used_names = set()
    final_cluster_map = {}
    for c_id in sorted(cluster_name_map.keys()):
        base_name = cluster_name_map[c_id]
        if base_name in used_names:
            final_name = f"{base_name} (Group {c_id + 1})"
        else:
            final_name = base_name
        used_names.add(final_name)
        final_cluster_map[c_id] = final_name

    print("K-Means Cluster Centroids Profile:")
    print(cluster_profiles)
    print("Assigned Business Cluster Labels:", final_cluster_map)

    # -------------------------------------------------------------
    # 4. Isolation Forest (Unusual / Anomalous Supplier Patterns)
    # -------------------------------------------------------------
    print("\n--- Training Isolation Forest Anomaly Detector ---")
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.045,  # Top ~4.5% anomalous patterns
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    anomaly_preds = iso_forest.fit_predict(X_all_scaled)
    # decision_function: lower means more abnormal
    anomaly_scores = iso_forest.decision_function(X_all_scaled)
    anomaly_flags = (anomaly_preds == -1).astype(int)

    anomalous_count = int(anomaly_flags.sum())
    print(f"Isolation Forest identified {anomalous_count} anomalous supplier operational patterns.")

    # -------------------------------------------------------------
    # Generate Predictions for Entire Dataset
    # -------------------------------------------------------------
    df["RF_Predicted_Risk_Category"] = rf_model.predict(X)
    rf_all_probs = rf_model.predict_proba(X)
    classes = list(rf_model.classes_)
    high_idx = classes.index("High Risk") if "High Risk" in classes else -1
    df["RF_High_Risk_Probability"] = [round(float(p[high_idx]), 4) if high_idx >= 0 else 0.0 for p in rf_all_probs]

    df["XGB_Predicted_Risk_Score"] = np.clip(xgb_model.predict(X), 0.0, 100.0).round(1)
    df["Supplier_Cluster"] = cluster_labels
    df["Supplier_Cluster_Label"] = [final_cluster_map[c] for c in cluster_labels]
    df["Anomaly_Flag"] = anomaly_flags
    df["Anomaly_Score"] = np.round(anomaly_scores, 4)
    df["Anomaly_Status"] = np.where(df["Anomaly_Flag"] == 1, "Anomaly Flagged", "Normal Operational")

    # Save modeled dataset
    modeled_file = DATA_DIR / "supplier_master_modeled.csv"
    df.to_csv(modeled_file, index=False)
    print(f"Saved modeled dataset with all 4 model outputs to: {modeled_file}")

    # -------------------------------------------------------------
    # Save Model Artifacts & Training Metrics
    # -------------------------------------------------------------
    joblib.dump(rf_model, MODELS_DIR / "rf_risk_classifier.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "xgb_risk_predictor.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans_segmentation.joblib")
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest_anomaly.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    metrics_payload = {
        "random_forest": {
            "accuracy": round(rf_acc, 4),
            "macro_f1": round(rf_f1_macro, 4),
            "classification_report": rf_report,
            "confusion_matrix": rf_cm,
            "labels": ["Low Risk", "Medium Risk", "High Risk"],
            "feature_importance": rf_importance_sorted
        },
        "xgboost": {
            "r2_score": round(xgb_r2, 4),
            "rmse": round(xgb_rmse, 4),
            "mae": round(xgb_mae, 4),
            "feature_importance": xgb_importance_sorted
        },
        "kmeans": {
            "n_clusters": 4,
            "cluster_labels_map": {str(k): v for k, v in final_cluster_map.items()},
            "cluster_counts": {str(k): int(v) for k, v in pd.Series(cluster_labels).value_counts().items()}
        },
        "isolation_forest": {
            "anomalies_detected": anomalous_count,
            "contamination_rate": 0.045
        }
    }

    metrics_file = MODELS_DIR / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_file}")

    return metrics_payload

if __name__ == "__main__":
    train_all_models()
