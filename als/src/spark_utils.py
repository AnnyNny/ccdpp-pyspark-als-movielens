# Spark session helper for the ALS 
import os, sys

if "JAVA_HOME" not in os.environ:
    _default_java = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
    if os.path.isdir(_default_java):
        os.environ["JAVA_HOME"] = _default_java

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession


def get_spark_session(app_name, num_cores, shuffle_partitions=8, driver_memory=None):
    # driver_memory ("6g") must be set before the JVM starts, useful for the
    # larger datasets (10M/32M)
    if driver_memory:
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-memory {driver_memory} pyspark-shell"

    spark = (
        SparkSession.builder
        .master(f"local[{num_cores}]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    return spark


def stop_spark(spark):

    if spark is not None:
        spark.catalog.clearCache()
        spark.stop()
