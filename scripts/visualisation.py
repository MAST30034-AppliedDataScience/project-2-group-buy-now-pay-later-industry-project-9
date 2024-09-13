from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_top_merchants_pie(top_n_merchants, level):
    """Plot a pie chart for the top N merchants by total revenue."""
    if not top_n_merchants.empty:
        plt.figure(figsize=(8, 8))
        plt.pie(top_n_merchants["total_revenue"], labels=top_n_merchants["name"], autopct='%1.1f%%', startangle=140)
        plt.title(f"Top 15 Companies by Total Revenue in Revenue Level '{level}'")
        plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        plt.show()
