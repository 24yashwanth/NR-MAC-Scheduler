import pandas as pd
import numpy as np
import sys


from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

schema = pq.read_schema("./Extra Files/BERLIN_V2X/sources/mobile_insight/pc1/LTE_PHY_PUSCH_Tx_Report.parquet", memory_map=True)
schema = pd.DataFrame(({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
schema = schema.reindex(columns=["column", "pa_dtype"], fill_value=pd.NA)  # Ensures columns in case the parquet file has an empty dataframe.

print(schema)

# table = pq.read_table('./Extra Files/BERLIN_V2X/sources/mobile_insight/pc1/LTE_PHY_PUSCH_Tx_Report.parquet')
# print(table['Serving Cell ID'])
