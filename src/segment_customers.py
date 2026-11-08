from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "customer_master_cleaned.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)
# Exclude names and IDs: they are identifiers, not meaningful clustering features.
# Exclude coordinates until their geographic validity is confirmed.
features = ["Industry", "Market_Segment", "Country", "City"]
prep = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), features)
])
model = Pipeline([
    ("prep", prep),
    ("cluster", KMeans(n_clusters=4, n_init=20, random_state=42))
])
df["customer_cluster"] = model.fit_predict(df[features])
joblib.dump(model, MODELS / "customer_kmeans.joblib")
df.to_csv(ROOT / "data" / "customer_master_segmented.csv", index=False)
print("Customer clusters created. Review cluster profiles before assigning business labels.")
