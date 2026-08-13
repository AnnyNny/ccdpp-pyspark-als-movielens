# ALS baseline using pyspark.ml (no manual ALS implementation).
# Fits the model, times training, evaluates test RMSE, and returns one
# result row in the standard schema (see results_utils.RESULT_COLUMNS).
from time import perf_counter

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

from results_utils import make_result

# default hyper-parameters
K = 5
LAMBDA = 0.1
MAX_ITER = 10
NUM_BLOCKS = 4
SEED = 42


def build_als(k=K, lam=LAMBDA, max_iter=MAX_ITER, num_blocks=NUM_BLOCKS, seed=SEED):
    """Construct an ALS estimator with the standard columns and options."""
    return ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        rank=k,
        regParam=lam,
        maxIter=max_iter,
        numUserBlocks=num_blocks,
        numItemBlocks=num_blocks,
        coldStartStrategy="drop",
        seed=seed,
    )


def run_als(train, test, dataset, dataset_size, num_cores,
            k=K, lam=LAMBDA, max_iter=MAX_ITER, num_blocks=NUM_BLOCKS,
            seed=SEED, warmup=True, run_id=None):
    """
    Fit ALS on `train`, evaluate test RMSE on `test`, and return one result row.

    train/test        : Spark DataFrames with columns user_id, item_id, rating
    dataset           : dataset name for the results row (e.g. "ml-1m")
    dataset_size      : total number of ratings in the dataset
    num_cores         : cores used for this run (recorded, not set here)
    warmup            : run one untimed fit first (JVM/JIT warm-up) so the
                        timed run reflects steady-state performance
    """
    als = build_als(k=k, lam=lam, max_iter=max_iter, num_blocks=num_blocks, seed=seed)
    evaluator = RegressionEvaluator(
        labelCol="rating", predictionCol="prediction", metricName="rmse"
    )

    if warmup:
        als.fit(train)

    start = perf_counter()
    model = als.fit(train)
    train_time = perf_counter() - start

    start = perf_counter()
    predictions = model.transform(test)
    test_rmse = evaluator.evaluate(predictions)
    eval_time = perf_counter() - start

    train_size = train.count()
    test_size = test.count()

    return make_result(
        algorithm="ALS",
        dataset=dataset,
        dataset_size=dataset_size,
        num_cores=num_cores,
        k=k,
        lam=lam,
        seed=seed,
        max_iter=max_iter,
        train_size=train_size,
        test_size=test_size,
        train_time=train_time,
        eval_time=eval_time,
        test_rmse=test_rmse,
        run_id=run_id,
    )
