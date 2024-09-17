import holidays
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType


def is_holiday(date):
    """
    Get all holidays in Australia in 2021
    """
    country = 'AU'
    year = 2021
    holiday_calendar = holidays.CountryHoliday(country, years=[year])
    return date in holiday_calendar


def is_special_date(df):
    """
    Check whether the date is a holiday and return the corresponding day of the week 
    for the given date in the DataFrame.
    """

    is_holiday_udf = F.udf(is_holiday, BooleanType())

    # Add a column to check if 'order_date' is a holiday
    df = df.withColumn("is_holiday", is_holiday_udf(F.col("order_datetime")))

    # Determine day of the week
    df = df.withColumn("day_of_week", F.date_format("order_datetime", "EEEE"))

    # Check if it is a weekend
    df = df.withColumn("is_weekend", F.date_format("order_datetime", "EEEE").isin("Saturday", "Sunday"))
