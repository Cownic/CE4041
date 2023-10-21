import numpy as np
import pandas as pd
import gc  # garbage collection
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import random

def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, dtype={
        'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str
    })

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop


####################### BLACK BOX IDK WHAT THEY DOING HERE################################
def get_landuse_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby("propertycountylandusecode")[
        "propertycountylandusecode"
    ].count()
    landuse_codes = list(temp[temp >= 300].index)
    temp = prop_2017.groupby("propertycountylandusecode")[
        "propertycountylandusecode"
    ].count()
    landuse_codes += list(temp[temp >= 300].index)
    landuse_codes = list(set(landuse_codes))
    df_landuse_codes = pd.DataFrame(
        {
            "propertycountylandusecode": landuse_codes,
            "propertycountylandusecode_id": range(len(landuse_codes)),
        }
    )
    return df_landuse_codes


def get_zoning_desc_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby("propertyzoningdesc")["propertyzoningdesc"].count()
    zoning_codes = list(temp[temp >= 5000].index)
    temp = prop_2017.groupby("propertyzoningdesc")["propertyzoningdesc"].count()
    zoning_codes += list(temp[temp >= 5000].index)
    zoning_codes = list(set(zoning_codes))
    df_zoning_codes = pd.DataFrame(
        {
            "propertyzoningdesc": zoning_codes,
            "propertyzoningdesc_id": range(len(zoning_codes)),
        }
    )
    return df_zoning_codes


def process_columns(df, df_landuse_codes, df_zoning_codes):
    df = df.merge(how="left", right=df_landuse_codes, on="propertycountylandusecode")
    df = df.drop(["propertycountylandusecode"], axis=1)

    df = df.merge(how="left", right=df_zoning_codes, on="propertyzoningdesc")
    df = df.drop(["propertyzoningdesc"], axis=1)

    df.loc[df.regionidcounty == 3101, "regionidcounty"] = 0
    df.loc[df.regionidcounty == 1286, "regionidcounty"] = 1
    df.loc[df.regionidcounty == 2061, "regionidcounty"] = 2

    df.loc[df.propertylandusetypeid == 279, "propertylandusetypeid"] = 261
    return df


####################### BLACK BOX IDK WHAT THEY DOING HERE################################


def feature_engineering(df):
    # Average Garage Size
    df["avg_garage_size"] = df["garagetotalsqft"] / df["garagecarcnt"]

    # Tax per square feet of the property
    df["property_tax_per_sqft"] = df["taxamount"] / df["calculatedfinishedsquarefeet"]

    # Rotated Coordinates of Property
    df["coord_1"] = df["latitude"] + df["longitude"]
    df["coord_2"] = df["latitude"] - df["longitude"]
    df["coord_3"] = df["latitude"] + 0.25 * df["longitude"]
    df["coord_4"] = df["latitude"] - 0.25 * df["longitude"]

    # Some of the features have repeated values as other exisiting columns
    # We change these columns to binary representation for whether they have value in that column or not and drop the original columns
    df["missing_finished_area"] = df["finishedsquarefeet12"].isnull().astype(np.float32)
    df["missing_total_area"] = df["finishedsquarefeet15"].isnull().astype(np.float32)
    df.drop(["finishedsquarefeet12", "finishedsquarefeet15"], axis=1, inplace=True)

    df["missing_bathroom_cnt_calc"] = (
        df["calculatedbathnbr"].isnull().astype(np.float32)
    )
    df.drop(["calculatedbathnbr"], axis=1, inplace=True)

    # Total Room Count in the property
    df["total_room_cnt"] = df["bedroomcnt"] + df["bathroomcnt"]

    # Average area in sqft per room
    mask = df.roomcnt >= 1  # avoid dividing by zero
    df.loc[mask, "avg_area_per_room"] = (
        df.loc[mask, "calculatedfinishedsquarefeet"] / df.loc[mask, "roomcnt"]
    )

    # Use the derived room_cnt to calculate the avg area again
    mask = df.total_room_cnt >= 1
    df.loc[mask, "derived_avg_area_per_room"] = (
        df.loc[mask, "calculatedfinishedsquarefeet"] / df.loc[mask, "total_room_cnt"]
    )

    return df


# Retype certain columns based on the input list to cetegorical columns and reducing columns from float64 to float32 for optimisation
def retype_columns(prop, feature_list):
    # Convert categorical variables to 'category' type, and float64 variables to float32
    for col in prop.columns:
        if col in feature_list:
            float_to_categorical(prop, col)
        elif prop[col].dtype.name == "float64":
            prop[col] = prop[col].astype(np.float32)
    return prop


# Convert columns to be of categorical type
def float_to_categorical(df, col):
    df[col] = df[col] - df[col].min()
    df.loc[df[col].isnull(), col] = -1
    df[col] = df[col].astype(int).astype("category")


def add_ymq_features(df):
    dt = pd.to_datetime(df.transactiondate).dt
    df["year"] = (dt.year - 2016).astype(int)
    df["month"] = (dt.month).astype(int)
    df["quarter"] = (dt.quarter).astype(int)
    df.drop(["transactiondate"], axis=1, inplace=True)
    return df


def save_models(models, modelname):
    for i, model in enumerate(models):
        model.save_model("checkpoints/" + modelname + "_" + str(i))
    print("Saved {} models to files.".format(len(models)))


def load_catboost_models(paths):
    models = []
    for path in paths:
        model = CatBoostRegressor()
        model.load_model(path)
        models.append(model)
    return models


def load_lightgbm_models(paths):
    models = []
    for path in paths:
        model = lgb.Booster(model_file=path)
        models.append(model)
    return models


def drop_features(df):
    # id and label (not features)
    unused_feature_list = ["parcelid", "logerror"]

    # too many missing
    missing_list = [
        "buildingclasstypeid",
        "architecturalstyletypeid" ,
        "storytypeid",
        "finishedsquarefeet13",
        "basementsqft",
        "yardbuildingsqft26",
    ]
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = [
        "fireplaceflag",
        "decktypeid",
        "pooltypeid10",
        "typeconstructiontypeid",
        "fips",
        "regionidcounty",
    ]
    unused_feature_list += bad_feature_list

    # hurts performance
    unused_feature_list += ['propertycountylandusecode_id', 'propertyzoningdesc_id']

    return df.drop(unused_feature_list, axis=1, errors="ignore")

def get_categorical_indices(df , feature_list):
    categorical_indexes = []
    for i, n in enumerate(df.columns):
        if n in feature_list:
            categorical_indexes.append(i)
    print(categorical_indexes)
    return categorical_indexes


def remove_outliers(X, y):
    outlier_threshold = 0.4
    mask = abs(y) <= outlier_threshold
    new_X = X[mask, :]
    new_y = y[mask]
    print("new_X: {}".format(new_X.shape))
    print("new_y: {}".format(new_y.shape))
    return new_X, new_y


"""
    Helper method that prepares 2016 and 2017 properties features for inference
"""


def transform_test_features(features_2016, features_2017):
    test_features_2016 = drop_features(features_2016)
    test_features_2017 = drop_features(features_2017)

    test_features_2016["year"] = 0
    test_features_2017["year"] = 1

    # 11 & 12 lead to unstable results, probably due to the fact that there are few training examples for them
    test_features_2016["month"] = 10
    test_features_2017["month"] = 10

    test_features_2016["quarter"] = 4
    test_features_2017["quarter"] = 4

    return test_features_2016, test_features_2017


"""
    Helper method that makes predictions on the test set and exports results to csv file
    'models' is a list of models for ensemble prediction (len=1 means using just a single model)
"""


def predict_and_export(models, features_2016, features_2017, file_name):
    # Construct DataFrame for prediction results
    submission_2016 = pd.DataFrame()
    submission_2017 = pd.DataFrame()
    submission_2016["ParcelId"] = features_2016.parcelid
    submission_2017["ParcelId"] = features_2017.parcelid

    test_features_2016, test_features_2017 = transform_test_features(
        features_2016, features_2017
    )

    pred_2016, pred_2017 = [], []
    for i, model in enumerate(models):
        print("Start model {} (2016)".format(i))
        pred_2016.append(
            #model.predict(test_features_2016, predict_disable_shape_check=True)
            model.predict(test_features_2016)
        )
        print("Start model {} (2017)".format(i))
        pred_2017.append(
            #model.predict(test_features_2017, predict_disable_shape_check=True)
            model.predict(test_features_2017)
        )

    # Take average across all models
    mean_pred_2016 = np.mean(pred_2016, axis=0)
    mean_pred_2017 = np.mean(pred_2017, axis=0)

    submission_2016["201610"] = [float(format(x, ".4f")) for x in mean_pred_2016]
    submission_2016["201611"] = submission_2016["201610"]
    submission_2016["201612"] = submission_2016["201610"]

    submission_2017["201710"] = [float(format(x, ".4f")) for x in mean_pred_2017]
    submission_2017["201711"] = submission_2017["201710"]
    submission_2017["201712"] = submission_2017["201710"]

    submission = submission_2016.merge(
        how="inner", right=submission_2017, on="ParcelId"
    )

    print("Length of submission DataFrame: {}".format(len(submission)))
    print("Submission header:")
    print(submission.head())
    submission.to_csv(file_name, index=False)
    return (
        submission,
        pred_2016,
        pred_2017,
    )  # Return the results so that we can analyze or sanity check it
