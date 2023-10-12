import numpy as np
import pandas as pd
import gc  # garbage collection


def load_data(file_path):
    return pd.read_csv(file_path)


def print_percent_missing(df):
    missing_percent = []
    total_count = len(df)
    for col in df.columns:
        missing_count = (df[col].isnull()).sum() + (df[col] == -1).sum()
        sett = (col , (missing_count/total_count)*100)
        missing_percent.append(sett)

    missing_percent.sort(key=lambda x:x[1])

    for feature, m_percent in missing_percent:
        print(f'{feature} : {m_percent}%\n')