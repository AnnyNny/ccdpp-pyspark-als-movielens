# Train/test splitting and preprocessing 

#   1. predefined split: MovieLens 100k ua.base/ua.test
#   2. hash-based split: seed=42, 10 bins, 1 bin = test
from pyspark.sql import functions as F


COLS = ["user_id", "item_id", "rating", "timestamp"]

SEED = 42
NUM_BINS = 10
TEST_BINS = 1  # 9 train / 1 test


def hash_split(ratings, seed=SEED, num_bins=NUM_BINS, test_bins=TEST_BINS):

    bucketed = ratings.withColumn(
        "split_bucket",
        F.pmod(F.xxhash64("user_id", "item_id", "timestamp", F.lit(seed)), F.lit(num_bins)),
    )
    threshold = num_bins - test_bins
    train = bucketed.filter(F.col("split_bucket") < threshold).drop("split_bucket")
    test = bucketed.filter(F.col("split_bucket") >= threshold).drop("split_bucket")
    return train, test


def clean_cold_start(train, test):
    """
    Drop test rows whose user or item never appears in train (cold-start),
    since ALS cannot score them
    """
    train_users = train.select("user_id").distinct()
    train_items = train.select("item_id").distinct()
    return (
        test
        .join(train_users, "user_id", "left_semi")
        .join(train_items, "item_id", "left_semi")
    )


def load_ml100k_predefined(spark, data_dir, base_file="ua.base", test_file="ua.test"):

    schema = "user_id INT, item_id INT, rating DOUBLE, timestamp LONG"
    train = spark.read.csv(f"{data_dir}/{base_file}", sep="\t", schema=schema)
    test = spark.read.csv(f"{data_dir}/{test_file}", sep="\t", schema=schema)
    return train, test
