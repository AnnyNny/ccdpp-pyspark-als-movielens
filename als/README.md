# ALS baseline

ALS (from `pyspark.ml`) used as a baseline to compare against CCD++.
This folder has the shared code (`src/`) and the experiment notebooks (`notebooks/`).

## What is here

`src/`
- `spark_utils.py` - make a local Spark session with a fixed number of cores.
- `datasets.py` - download a MovieLens dataset, load the ratings, and make the train/test split as Parquet.
- `split_utils.py` - the split logic (hash split, cold-start cleaning, and the 100k predefined split).
- `als_utils.py` - build and run ALS, return one result row.
- `results_utils.py` - the columns of a result row and how to append it to the CSV.

`notebooks/`
- `01_als_100k.ipynb` - MovieLens 100k, uses the official `ua.base` / `ua.test` split.
- `02_als_1m.ipynb`, `03_als_10m.ipynb`, `04_als_32m.ipynb` - the larger sets, hash split saved to Parquet.
- `05_als_tuning.ipynb` - runs a grid of k and lambda values on 100k to see their effect on RMSE.

The notebooks are thin: they set the parameters and call the functions in `src/`.


## The split

- 100k: keep the split that comes with the dataset (`ua.base` / `ua.test`).
- Everything else: hash each rating into 10 bins (seed 42), 9 bins for train and 1 for test.
  Test rows with a user or item not seen in
  train are dropped, because ALS can't score them.

## Results

Every run adds one line to `results/als_results.csv` with the same columns:
algorithm, dataset, sizes, cores, k, lambda, seed, maxIter, train/eval/total time,
test RMSE, and a run id. This makes it easy to compare ALS and CCD++ later.

## How to run

You need Java (17 or newer) and Python. Install the packages once:

    pip install -r requirements.txt

Then open a notebook, pick the parameters at the top, and run all cells.
The dataset is downloaded the first time and reused after that.

### Note on Windows

Reading and running ALS works on Windows. Writing Parquet does not, because Spark
needs some Hadoop Windows files that are not included. So:

- 100k runs fine on Windows (it does not write Parquet).
- 1M / 10M / 32M write Parquet, so run them on Linux. WSL (Ubuntu) works.
  In WSL, make a separate venv and `pip install -r requirements.txt` there.

## Cores and memory

Always pass the number of cores explicitly (`get_spark_session(name, num_cores=...)`),
so a run never depends on a Spark default. For the big sets you can give the driver
more memory, for example `driver_memory="6g"`.
