import numpy as np
import pandas as pd
import gc  # garbage collection


def load_data(file_path):
    return pd.read_csv(file_path)


def merge_data(df1, df2, on_v):
    return pd.merge(df1, df2, how='left' , on=on_v)

def drop_dups(df):
    # We are keeping the last entry for those repeated ones
    return df.drop_duplicates(subset=['parcelid' , 'transactiondate'], ignore_index=True, keep='last')

def check_duplicates(housing): 
    idsUnique = len(housing[['parcelid', 'transactiondate']].value_counts())
    idsTotal = housing.shape[0]
    idsDupli = idsTotal - idsUnique
    print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

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

