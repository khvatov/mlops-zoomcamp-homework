import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    current_dir = Path('.').resolve()
    mlflow_db=(Path(current_dir) / 'mlflow.db').resolve()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("nyc-taxi-experiment1")

    mlflow.autolog()


    with mlflow.start_run():
    #     mlflow.set_tag("model", "xgboost")
        #mlflow.log_params({'max_depth':10, 'random_state':0})
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        # mlflow.log_metric("rmse", rmse)
        #mlflow.lingreg.log_model(rf, artifact_path="models_mlflow")

if __name__ == '__main__':
    run_train()