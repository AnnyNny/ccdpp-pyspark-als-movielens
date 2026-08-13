
import csv
import os
from datetime import datetime


RESULT_COLUMNS = [
    "run_id",          # unique id / timestamp for the run
    "timestamp",       # ISO time the row was written
    "algorithm",       # e.g. "ALS"
    "dataset",         # e.g. "ml-100k", "ml-1m", "ml-32m"
    "dataset_size",    # total number of ratings in the dataset
    "num_cores",       # explicit Spark cores used
    "k",               # rank
    "lambda",          # regParam
    "seed",
    "max_iter",
    "train_size",
    "test_size",
    "train_time",      # seconds spent in fit()
    "eval_time",       # seconds spent evaluating (0 if not measured)
    "total_time",      # train_time + eval_time
    "test_rmse",
]


def new_run_id():
    """Timestamp-based id, e.g. 20260813_142530_123456."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def make_result(algorithm, dataset, dataset_size, num_cores, k, lam, seed,
                max_iter, train_size, test_size, train_time, test_rmse,
                eval_time=0.0, run_id=None):
    
    if run_id is None:
        run_id = new_run_id()
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "dataset": dataset,
        "dataset_size": dataset_size,
        "num_cores": num_cores,
        "k": k,
        "lambda": lam,
        "seed": seed,
        "max_iter": max_iter,
        "train_size": train_size,
        "test_size": test_size,
        "train_time": train_time,
        "eval_time": eval_time,
        "total_time": train_time + eval_time,
        "test_rmse": test_rmse,
    }


def append_result(row, csv_path):
   
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
