import pandas as pd
import numpy as np
from time import *

start_time = time()
# Read Furnace Data
FurnaceData = pd.read_csv("00Furnace_Data.csv")

# Delete Null Data
FurnaceData.dropna(axis=0, how='any', inplace=True)

# Read Data Column name
FurnaceData_ColumnList = FurnaceData.columns.tolist()

# Data Clean
for column_name in FurnaceData_ColumnList:
    column_number = FurnaceData_ColumnList.index(column_name)
    indexes = []
    for row_number, i in enumerate(FurnaceData.index):
        # Delete 0 Value
        if FurnaceData.iat[row_number, column_number] == 0.0:
            indexes.append(i)
    FurnaceData.drop(indexes, axis=0, inplace=True)
    # Remove Duplicate Data
    if column_name != "FEED":
        FurnaceData.drop_duplicates(column_name, inplace=True)
    # Delete Outlier Data
    FurnaceData = FurnaceData[
        np.abs(FurnaceData[column_name] - FurnaceData[column_name].mean()) <= (
                3 * FurnaceData[column_name].std()
        )
        ]

# Export Data to CSV File
FurnaceData.to_csv("00FurnaceCleanData.csv")
FurnaceData.describe().to_csv("00FurnaceCleanDataDescribe.csv")

# Run Time
finish_time = time()
run_time = finish_time - start_time
print("Total program running time: ", run_time)
