from pyspark.sql.dataframe import DataFrame as SparkDataFrame


def spark_load_table(table: str) -> SparkDataFrame:
    return spark.table(table)