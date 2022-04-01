from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from telco_churn.utils.get_spark import spark


def spark_load_table(table: str) -> SparkDataFrame:
    return spark.table(table)
