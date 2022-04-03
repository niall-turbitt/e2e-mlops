import pyspark

from telco_churn.utils.get_spark import spark


def spark_load_table(table: str) -> pyspark.sql.dataframe.DataFrame:
    return spark.table(table)
