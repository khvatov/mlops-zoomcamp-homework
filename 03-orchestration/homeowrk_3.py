#!/usr/bin/env python
# coding: utf-8

import mlflow.sklearn
import pandas as pd

import pickle
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error


from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment-hw3")


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    df = pd.read_parquet(url)
    print(f"Read {len(df)} rows for {year}-{month:02d}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Remove outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    print(f"After cleaning {len(df)} rows for {year}-{month:02d}")

    return df




def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv




def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
   
        print(f"Intercept: {lr.intercept_}")
        print(f"Coefficients: {lr.coef_}")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("num_train_samples", X_train.shape[0])
        mlflow.log_param("num_val_samples", X_val.shape[0])
        mlflow.log_metric("intercept", lr.intercept_)
    
    
        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            print("Saving DictVectorizer to models/preprocessor.b")
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

        return run.info.run_id

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

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    print(f"Model trained and logged for {year}-{month:02d} and {next_year}-{next_month:02d}")

    return run_id

if __name__ == "__main__":
    import argparse

    parsers = argparse.ArgumentParser(description="Train a duration prediction model for NYC taxi trips.")
    parsers.add_argument('--year', type=int, required=True, help='Year of the data to process')
    parsers.add_argument('--month', type=int, required=True, help='Month of the data to process')
    args = parsers.parse_args()

    run_id = run(args.year, args.month)


    #save run_id to a file
    with open("models/run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID saved to models/run_id.txt: {run_id}")
    print("Process completed successfully.")
    print("You can now run the model with the following command:")
    print(f"mlflow models serve -m models_mlflow -p 5001 --no-conda")
