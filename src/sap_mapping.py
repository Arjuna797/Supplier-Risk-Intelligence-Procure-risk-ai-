"""
SAP Materials Management (MM) Data Mapping & Schema Definitions.
Maps supplier procurement attributes to SAP tables and fields:
- Vendor Master: LFA1 (General), LFB1 (Company Code), LFM1 (Purchasing Org) / S/4HANA BUT000
- Purchasing Documents: EKKO (PO Header), EKPO (PO Item), EKBE (PO History)
- Vendor Evaluation: ME61 / ME64 / S_ALR_87013155
- Quality Management: QALS (Inspection Lot), QMEL (Notifications)
"""

SAP_FIELD_MAPPING = {
    "Supplier_ID": {
        "sap_table": "LFA1 / BUT000",
        "sap_field": "LIFNR",
        "description": "Vendor Account Number (Supplier Unique ID)",
        "type": "string",
        "required": True,
        "sample": "VEND-10024"
    },
    "Supplier_Name": {
        "sap_table": "LFA1 / BUT000",
        "sap_field": "NAME1",
        "description": "Name 1 of Supplier / Business Partner",
        "type": "string",
        "required": True,
        "sample": "Apex Precision Instruments"
    },
    "Country": {
        "sap_table": "LFA1",
        "sap_field": "LAND1",
        "description": "Country Key (ISO or Full Country Name)",
        "type": "string",
        "required": True,
        "sample": "Germany"
    },
    "City": {
        "sap_table": "LFA1",
        "sap_field": "ORT01",
        "description": "City of Supplier Registered Office / Plant",
        "type": "string",
        "required": True,
        "sample": "Stuttgart"
    },
    "Material_Group": {
        "sap_table": "EKPO / MARA",
        "sap_field": "MATKL",
        "description": "Material Group / Commodity Code",
        "type": "string",
        "required": True,
        "sample": "Electronics & Sensors"
    },
    "Purchasing_Org": {
        "sap_table": "LFM1 / EKKO",
        "sap_field": "EKORG",
        "description": "Purchasing Organization code",
        "type": "string",
        "required": False,
        "sample": "1000"
    },
    "Payment_Terms": {
        "sap_table": "LFM1 / EKKO",
        "sap_field": "ZTERM",
        "description": "Terms of Payment Key",
        "type": "string",
        "required": False,
        "sample": "NT30"
    },
    "PO_Count": {
        "sap_table": "EKKO",
        "sap_field": "COUNT(EBELN)",
        "description": "Total Purchase Orders executed",
        "type": "integer",
        "required": True,
        "sample": 142
    },
    "Total_Spend_USD": {
        "sap_table": "EKPO",
        "sap_field": "SUM(NETWR)",
        "description": "Total Annual Spend / Net Order Value in USD",
        "type": "float",
        "required": True,
        "sample": 1850000.00
    },
    "On_Time_Delivery_Rate": {
        "sap_table": "ME61 / EKBE",
        "sap_field": "DELIV_ONTIME_PCT",
        "description": "Percentage of deliveries on or before scheduled date (0-100)",
        "type": "float",
        "required": True,
        "sample": 94.5
    },
    "In_Full_Delivery_Rate": {
        "sap_table": "ME61 / EKBE",
        "sap_field": "DELIV_INFULL_PCT",
        "description": "Percentage of deliveries fulfilling 100% of PO quantity (0-100)",
        "type": "float",
        "required": True,
        "sample": 96.0
    },
    "Defect_Rate_Pct": {
        "sap_table": "QALS / ME61",
        "sap_field": "DEFECT_PCT",
        "description": "Goods receipt quality rejection / defect percentage (0-100)",
        "type": "float",
        "required": True,
        "sample": 1.25
    },
    "Average_Delay_Days": {
        "sap_table": "EKET / EKBE",
        "sap_field": "AVG_DELAY",
        "description": "Average delay in days beyond committed delivery date",
        "type": "float",
        "required": True,
        "sample": 1.8
    },
    "Lead_Time_Days": {
        "sap_table": "EKPO",
        "sap_field": "PLIFZ",
        "description": "Planned / Actual Delivery Time in calendar days",
        "type": "float",
        "required": True,
        "sample": 28.0
    },
    "Price_Variance_Pct": {
        "sap_table": "RSEG / EKPO",
        "sap_field": "PRICE_VAR_PCT",
        "description": "Invoice vs PO Price variance percentage",
        "type": "float",
        "required": True,
        "sample": 0.8
    },
    "Financial_Stability_Score": {
        "sap_table": "Ext / BP",
        "sap_field": "CREDIT_RATING",
        "description": "Credit score / Financial solvency rating (1-100, 100=highest)",
        "type": "float",
        "required": True,
        "sample": 82.0
    },
    "Geopolitical_Risk_Score": {
        "sap_table": "Ext / Country",
        "sap_field": "GEO_RISK",
        "description": "Country / Regional risk index (1-100, 100=highest risk)",
        "type": "float",
        "required": True,
        "sample": 18.0
    },
    "Single_Source_Flag": {
        "sap_table": "ME01 / EORD",
        "sap_field": "SOLE_SUPPLIER",
        "description": "Binary flag (1 if sole supplier for material group, else 0)",
        "type": "integer",
        "required": True,
        "sample": 0
    },
    "Contract_Compliance_Rate": {
        "sap_table": "EKKO",
        "sap_field": "KONNR_COMPLIANCE",
        "description": "Percentage of orders referencing master agreement (0-100)",
        "type": "float",
        "required": False,
        "sample": 95.0
    },
    "ESG_Score": {
        "sap_table": "Ext / BP",
        "sap_field": "ESG_RATING",
        "description": "Environmental, Social and Governance index (1-100, 100=best)",
        "type": "float",
        "required": False,
        "sample": 88.0
    }
}

REQUIRED_COLUMNS = [col for col, meta in SAP_FIELD_MAPPING.items() if meta["required"]]

def get_field_mapping_dataframe():
    """Return a clean pandas DataFrame summarizing the SAP MM schema mapping."""
    import pandas as pd
    rows = []
    for col, meta in SAP_FIELD_MAPPING.items():
        rows.append({
            "CSV Column": col,
            "SAP Table": meta["sap_table"],
            "SAP Field / Transaction": meta["sap_field"],
            "Description": meta["description"],
            "Data Type": meta["type"],
            "Required": "Yes" if meta["required"] else "Optional",
            "Sample Value": str(meta["sample"])
        })
    return pd.DataFrame(rows)
