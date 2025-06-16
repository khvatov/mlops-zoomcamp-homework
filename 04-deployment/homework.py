import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path


categorical = ['PULocationID', 'DOLocationID']


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
        return dv, model

def read_data(filename):
    
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df, model, dv):

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def run(year, month):
    dv, model = load_model()
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    y_pred = predict(df, model, dv)
    std_dev = np.std(y_pred)
    print(f'Standard diviation: {std_dev:.4f}')


    mean = np.mean(y_pred)
    print(f'Mean {mean:.3f}')

    print("Saving results to file...")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction']= y_pred

    output_file = Path('.') / 'output' / f'yellow_tripdata_predictions_{year:04d}-{month:02d}.parquet'
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python homework.py <year> <month>')
        sys.exit(1)
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    run(year, month)
    