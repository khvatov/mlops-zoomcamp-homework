#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import xgboost as xgb
import pickle
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    

    return df




def create_X(df, dv=None):
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)


    # train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    # X_train = dv.fit_transform(train_dicts)

    # val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    # X_val = dv.transform(val_dicts)
    return X, dv




def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            print("Saving DictVectorizer to models/preprocessor.b")
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

def run(year, month):
    

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_train = read_dataframe(year, month)
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)


    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":
    import argparse

    parsers = argparse.ArgumentParser(description="Train a duration prediction model for NYC taxi trips.")
    parsers.add_argument('--year', type=int, required=True, help='Year of the data to process')
    parsers.add_argument('--month', type=int, required=True, help='Month of the data to process')
    args = parsers.parse_args()

    run(args.year, args.month)
