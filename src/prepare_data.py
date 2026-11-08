from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "customer_master.csv"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SOURCE)
required = ["Customer_ID", "Customer_Name", "Industry", "Market_Segment",
            "Country", "City", "Latitude", "Longitude"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# Preserve original fields; add quality flags rather than silently altering source values.
df["duplicate_id_flag"] = df["Customer_ID"].duplicated(keep=False).astype(int)
df["missing_value_count"] = df[required].isna().sum(axis=1)
df["invalid_latitude_flag"] = (~df["Latitude"].between(-90, 90)).astype(int)
df["invalid_longitude_flag"] = (~df["Longitude"].between(-180, 180)).astype(int)

# A city-country reference table is intentionally explicit and editable.
# This checks only the cities listed here; it is not a comprehensive world gazetteer.
city_country = {
    "London": {"UK"}, "Paris": {"France"}, "Berlin": {"Germany"},
    "Madrid": {"Spain"}, "Rome": {"Italy"}, "Tokyo": {"Japan"},
    "Mumbai": {"India"}, "Sydney": {"Australia"}, "New York": {"USA"},
    "Toronto": {"Canada"}, "Copenhagen": {"Denmark"}, "Seoul": {"South Korea"},
    "Singapore": {"Singapore"}, "Amsterdam": {"Netherlands"}, "Zurich": {"Switzerland"},
    "Sao Paulo": {"Brazil"}
}
df["city_country_check"] = df.apply(
    lambda r: "Unmapped city" if r["City"] not in city_country
    else ("Review mismatch" if r["Country"] not in city_country[r["City"]] else "Consistent"),
    axis=1
)

df["data_quality_flag"] = (
    (df["duplicate_id_flag"] == 1)
    | (df["missing_value_count"] > 0)
    | (df["invalid_latitude_flag"] == 1)
    | (df["invalid_longitude_flag"] == 1)
    | (df["city_country_check"] != "Consistent")
).astype(int)

df.to_csv(OUT / "customer_master_cleaned.csv", index=False)
print(f"Processed {len(df)} customer records.")
print("Quality flags:", int(df["data_quality_flag"].sum()))
print(f"Output: {OUT / 'customer_master_cleaned.csv'}")
