from pyspark.sql import functions as F, SparkSession, DataFrame
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType, DoubleType
from functools import reduce
import zipfile
import os

def replace_id(map_df, target_df):
    """
        Replace all user_id by consumer_id
    """
    mapped_df = target_df.join(map_df, on="user_id", how="inner")
    mapped_df = mapped_df.drop('user_id')
    
    return mapped_df

def load_merchant_details(merchant_df):
    
    """
        This function load data on merchants info and transform it into a suitable format
    """
    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"^[\(\[]|[\)\]]$", "")) # Remove the outermost bracket
    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"[\)\]],\s*[\(\[]", r")\|(")) # Replacing the comma that seperate each touple/list into "|"

    # Split accorddingly 
    merchant_df = merchant_df.withColumn("tags", F.split("tags", "\|")) 
    merchant_df = merchant_df.withColumns({"category": F.regexp_replace(F.col("tags").getItem(0), r"^[\(\[]|[\)\]]$", ""),
                         "revenue_level": F.regexp_replace(F.col("tags").getItem(1), r"^[\(\[]|[\)\]]$", ""),
                         "take_rate": F.regexp_extract(F.col("tags").getItem(2), r"take rate: (\d+\.\d+)",1).cast(DoubleType())
                        })
    
    merchant_df = merchant_df.withColumn("category", F.lower(F.col("category")))

    merchant_df = merchant_df.drop("tags")

    merchant_df = merchant_df.filter((F.col("revenue_level") == "a") | (F.col("revenue_level") == "b") | (F.col("revenue_level") == "c") |
                   (F.col("revenue_level") == "d") | (F.col("revenue_level") == "e"))
    
    return merchant_df


# Restructure tbl_consumer dataframe
def load_consumer_details(consumer_df):
    """
        This function restructure the `tabl_consumer.csv` data into a conformable format
    """
    column_name = str(consumer_df.columns[0])
    consumer_df = consumer_df.withColumn("name", F.split(consumer_df[column_name], r'\|').getItem(0)) \
                .withColumn("consumer_id", F.split(consumer_df[column_name], r'\|').getItem(5))\
                .withColumn("gender", F.split(consumer_df[column_name], r'\|').getItem(4))\
                .withColumn("state", F.split(consumer_df[column_name], r'\|').getItem(2)) \
                .withColumn("postcode", F.split(consumer_df[column_name], r'\|').getItem(3)) \
            

    consumer_df = consumer_df.withColumn("postcode", F.col("postcode").cast(IntegerType())) \
                .withColumn("consumer_id", F.col("consumer_id").cast(LongType()))
    consumer_df = consumer_df.drop(column_name)

    return consumer_df