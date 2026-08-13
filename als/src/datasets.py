
import urllib.request
import zipfile
from pathlib import Path

from pyspark.sql import functions as F

#metadata:
#   url     : download url
#   folder  : folder name created after extraction (note ml-10m -> ml-10M100K)
#   ratings : ratings file inside that folder
#   sep     : field separator ("::" for .dat, "," for .csv)
#   header  : whether the ratings file has a header row
MOVIELENS = {
    "ml-100k": {"url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
                "folder": "ml-100k",     "ratings": "u.data",      "sep": "\t", "header": False},
    "ml-1m":   {"url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                "folder": "ml-1m",       "ratings": "ratings.dat", "sep": "::", "header": False},
    "ml-10m":  {"url": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
                "folder": "ml-10M100K",  "ratings": "ratings.dat", "sep": "::", "header": False},
    "ml-20m":  {"url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
                "folder": "ml-20m",      "ratings": "ratings.csv", "sep": ",",  "header": True},
    "ml-32m":  {"url": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
                "folder": "ml-32m",      "ratings": "ratings.csv", "sep": ",",  "header": True},
}


def ensure_movielens(name, data_dir):
    
    data_dir = Path(data_dir)
    dataset_dir = data_dir / MOVIELENS[name]["folder"]
    if dataset_dir.is_dir():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(MOVIELENS[name]["url"], zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    return dataset_dir


def load_ratings(spark, name, data_dir):
    """
    Load a MovieLens ratings file into the standard schema
    (user_id, item_id, rating, timestamp)
    """
    info = MOVIELENS[name]
    dataset_dir = ensure_movielens(name, data_dir)
    path = str(dataset_dir / info["ratings"])

    if info["sep"] == "::":
        raw = spark.read.text(path)
        p = F.split(raw["value"], "::")
        return raw.select(
            p.getItem(0).cast("int").alias("user_id"),
            p.getItem(1).cast("int").alias("item_id"),
            p.getItem(2).cast("double").alias("rating"),
            p.getItem(3).cast("long").alias("timestamp"),
        )

    # CSV with header (ml-20m / ml-32m: userId,movieId,rating,timestamp).
    # Read by header name and rename to the standard schema
    raw = spark.read.option("header", "true").option("sep", info["sep"]).csv(path)
    return raw.select(
        raw["userId"].cast("int").alias("user_id"),
        raw["movieId"].cast("int").alias("item_id"),
        raw["rating"].cast("double").alias("rating"),
        raw["timestamp"].cast("long").alias("timestamp"),
    )


def prepare_split(spark, name, data_dir, processed_dir, seed=42, overwrite=False):
    """
    Preprocess a dataset with hash-split (seed, 10 bins, 9 train/1 test), then write train, test_clean to Parquet. 
    Returns (train_path, test_path) as strings.
    """
    from split_utils import hash_split, clean_cold_start

    out = Path(processed_dir) / name
    train_path = str(out / "train")
    test_path = str(out / "test_clean")

    if not overwrite and (out / "train" / "_SUCCESS").exists() \
            and (out / "test_clean" / "_SUCCESS").exists():
        return train_path, test_path

    ratings = load_ratings(spark, name, data_dir)
    train, test = hash_split(ratings, seed=seed)
    test_clean = clean_cold_start(train, test)

    train.write.mode("overwrite").parquet(train_path)
    test_clean.write.mode("overwrite").parquet(test_path)
    return train_path, test_path
