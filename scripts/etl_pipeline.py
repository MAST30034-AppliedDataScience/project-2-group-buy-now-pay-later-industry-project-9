from pyspark.sql import functions as F, SparkSession, DataFrame
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType, DoubleType
from functools import reduce
from pyspark.sql.functions import col, sum
import zipfile
import os

def replace_id(map_df, target_df):
    """
        Replace all user_id by consumer_id
    """
    mapped_df = target_df.join(map_df, on="user_id", how="inner")
    mapped_df = mapped_df.drop('user_id')
    
    return mapped_df

def clean_merchant_details(merchant_df):
    
    """
        This function takes in a merchants dataset and transforms/cleans it into a suitable format.
        Also indicates dataset size before and after. Returns cleaned dataset.
    """

    # Get dataset size before cleaning
    print("Before: ")
    get_dataset_count(merchant_df)

    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"^[\(\[]|[\)\]]$", "")) # Remove the outermost bracket
    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"[\)\]],\s*[\(\[]", r")\|(")) # Replacing the comma that separate each tuple/list into "|"

    # Split accordingly 
    merchant_df = merchant_df.withColumn("tags", F.split("tags", "\|")) 
    merchant_df = merchant_df.withColumns({"category": F.regexp_replace(F.col("tags").getItem(0), r"^[\(\[]|[\)\]]$", ""),
                         "revenue_level": F.regexp_replace(F.col("tags").getItem(1), r"^[\(\[]|[\)\]]$", ""),
                         "take_rate": F.regexp_extract(F.col("tags").getItem(2), r"take rate: (\d+\.\d+)",1).cast(DoubleType())
                        })
    
    # Make it consistently lower case
    merchant_df = merchant_df.withColumn("category", F.lower(F.col("category")))

    # Drop original feature column (not needed anymore)
    merchant_df = merchant_df.drop("tags")

    # Ensure revenue level is within a defined range of (a-e)
    merchant_df = merchant_df.filter((F.col("revenue_level") == "a") | (F.col("revenue_level") == "b") | (F.col("revenue_level") == "c") |
                   (F.col("revenue_level") == "d") | (F.col("revenue_level") == "e"))
    

    # Ensure take_rate is within a defined range (0.0 to 100.0)
    merchant_df = merchant_df.filter((F.col("take_rate") >= 0.0) & (F.col("take_rate") <= 100.0))

    # Get dataset size after cleaning
    print("After: ")
    get_dataset_count(merchant_df)

    return merchant_df


def clean_consumer_details(consumer_df):
    """
        This function takes in the consumer dataset, and restructures the data into a suitable format.
        Also indicates dataset size before and after. Returns cleaned dataset.
    """

    # Get dataset size before cleaning
    print("Before: ")
    get_dataset_count(consumer_df)


    column_name = str(consumer_df.columns[0])
    consumer_df = consumer_df.withColumn("name", F.split(consumer_df[column_name], r'\|').getItem(0)) \
                .withColumn("consumer_id", F.split(consumer_df[column_name], r'\|').getItem(5))\
                .withColumn("gender", F.split(consumer_df[column_name], r'\|').getItem(4))\
                .withColumn("state", F.split(consumer_df[column_name], r'\|').getItem(2)) \
                .withColumn("postcode", F.split(consumer_df[column_name], r'\|').getItem(3)) \
            

    consumer_df = consumer_df.withColumn("postcode", F.col("postcode").cast(IntegerType())) \
                .withColumn("consumer_id", F.col("consumer_id").cast(LongType()))
    consumer_df = consumer_df.drop(column_name)

    # Get dataset size after cleaning
    print("After: ")
    get_dataset_count(consumer_df)

    return consumer_df

def get_dataset_count(df):

    """
        This function takes in a dataset and prints its count. (size) 
    """

    count = df.count()
    print("The dataset count is ", count )

    return

def ensure_datetime_range(df, start, end):
    """
        This function ensures that a dataframe with a column that specifies datetime is within the desire datetime range
    """
    inital_entries = df.count()
    df = df.filter((start <= F.to_date(F.col("order_datetime"))) &
                           (F.to_date(F.col("order_datetime")) <= end))
    
    final_entries = df.count()
    print(f"Starting entries: {inital_entries} \nFinal entries: {final_entries}")
    print(f"Net change (%): {round((inital_entries - final_entries)/inital_entries * 100, 2)} ")
    return df

def calculate_missing_values(df):
    """
    Takes in a DataFrame, calculates the count of missing (NULL) values 
    for each column, and displays it as a table.
    """

    # Initialise an empty list to hold the expressions for summing NULL counts
    null_count_columns = []

    # Iterate over each column, summing their counts of NULL values
    for column in df.columns:
        null_count_expression = F.sum(F.col(column).isNull().cast("int")).alias(column + '_missing_count')
        null_count_columns.append(null_count_expression)

    # Select the summed NULL counts and display them
    missing_value_counts = df.select(null_count_columns)
    missing_value_counts.show()

    return
