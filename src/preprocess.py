from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "income","household_size","home_age","sqft",
    "baseline_gas_therms_year","baseline_elec_kwh_year",
    "elec_rate_cents_kwh","gas_rate_dollars_therm"
]

CATEGORICAL_COLS = [
    "building_type","region","utility","urban","dac_flag","received_prior_rebate"
]

TARGET_COL = "adopted_heat_pump"

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def build_preprocessor() -> ColumnTransformer:
    numeric_t = ("num", StandardScaler(), NUMERIC_COLS)
    categorical_t = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
    preprocessor = ColumnTransformer([numeric_t, categorical_t])
    return preprocessor

def get_feature_lists():
    return NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL
