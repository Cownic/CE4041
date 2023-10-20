#!/bin/python3
# run this file in it's current directory

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from helper import utility

DATA_DIR: str = "../data"

CATBOOST_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.05,
    'depth': 16,
    'verbose': 20,
    #     'l2_leaf_reg': 1000,
    'task_type': 'CPU',
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': 0,
}

N_ENSEMBLES = 8


def add_date_features(df: pd.DataFrame):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    # df.drop(["transactiondate"], inplace=True, axis=1)
    return df


def print_feature_importance(model: CatBoostRegressor, pool: Pool, X_train: pd.DataFrame):
    feature_importances = model.get_feature_importance(pool)
    feature_names = X_train.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}\t{}'.format(name, score))


# start here
data = pd.read_csv(f"{DATA_DIR}/train.csv", low_memory=False)
sample_submission = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv",
    low_memory=False
)
properties2016 = pd.read_csv(
    f'{DATA_DIR}/properties_2016.csv',
    low_memory=False
)

data_used = data.drop(columns=["logerror"])
# used when re-ordering cols for prediction
model_features = data_used.columns.tolist()

sample_submission["parcelid"] = sample_submission["ParcelId"]
test_df = pd.merge(
    sample_submission,
    properties2016,
    how='left',
    on='parcelid'
)

print(data)

X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame

X_train, X_test, y_train, y_test = train_test_split(
    data_used,
    data.logerror,
    test_size=0.2,
    random_state=99
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

all_pool = Pool(data_used, data.logerror)
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

model = CatBoostRegressor(**CATBOOST_PARAMS)
model.fit(train_pool, eval_set=test_pool)

print_feature_importance(model, train_pool, X_train)

submission = pd.DataFrame({
    'ParcelId': test_df['parcelid'],
})

test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}

# ensemble models
models: list[CatBoostRegressor] = [None] * N_ENSEMBLES
for i in range(N_ENSEMBLES):
    print("\nTraining (ensemble): %d ..." % (i))
    CATBOOST_PARAMS['random_seed'] = i
    models[i] = CatBoostRegressor(**CATBOOST_PARAMS)
    models[i].fit(train_pool, eval_set=test_pool)
    print('-- Feature Importance --')
    print_feature_importance(models[i], train_pool, X_train)


for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    test_df['transactiondate'] = test_date
    test_df = add_date_features(test_df)
    test_df = utility.add_dmy_feature(test_df)
    y_pred = 0.0
    for i in range(N_ENSEMBLES):
        print("Ensemble:", i)
        y_pred += models[i].predict(test_df[model_features])
    y_pred /= N_ENSEMBLES
    submission[label] = y_pred

print("Creating submission: submission ...")
submission.to_csv(
    f'{DATA_DIR}/submission_iter{CATBOOST_PARAMS["iterations"]}_depth{CATBOOST_PARAMS["depth"]}_l{CATBOOST_PARAMS["learning_rate"]}_e{N_ENSEMBLES}.csv',
    float_format='%.4f',
    index=False)
print("Finished.")
