import os, sys

os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
#os.environ["HADOOP_HOME"] = os.path.join(sys.prefix, "hadoop")

from pyspark.sql import SparkSession

def get_spark_session(app_name="Test", num_cores="*"):

    spark = (
        SparkSession.builder
        .master(f"local[{num_cores}]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    return spark