import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, StringType
)

ROOT = 'data/'
ANIME_ROUTE = ROOT + 'anime.csv'
RATING_COMPLETE_ROUTE = ROOT + 'rating_complete.csv'
RATING_EP_ROUTE = ROOT + 'valoraciones_EP.csv'
cpu_cores = multiprocessing.cpu_count()
spark = SparkSession.builder.master(f"local[{cpu_cores - 1}]").appName("Reviews").getOrCreate()

def eda_df_anime():
    schema = StructType([
        StructField("ID", IntegerType(), nullable=True),
        StructField("Name", StringType(), nullable=True),
        StructField("Score", DoubleType(), nullable=True),
        StructField("Genres", StringType(), nullable=True),
        StructField("English name", StringType(), nullable=True),
        StructField("Japanese name", StringType(), nullable=True),
        StructField("Type", StringType(), nullable=True),
        StructField("Episodes", IntegerType(), nullable=True),
        StructField("Aired", StringType(), nullable=True),
        StructField("Premiered", StringType(), nullable=True),
        StructField("Producers", StringType(), nullable=True),
        StructField("Licensors", StringType(), nullable=True),
        StructField("Studios", StringType(), nullable=True),
        StructField("Source", StringType(), nullable=True),
        StructField("Duration", StringType(), nullable=True),
        StructField("Rating", StringType(), nullable=True),
        StructField("Ranked", DoubleType(), nullable=True),
        StructField("Popularity", IntegerType(), nullable=True),
        StructField("Members", IntegerType(), nullable=True),
        StructField("Favorites", IntegerType(), nullable=True),
        StructField("Watching", IntegerType(), nullable=True),
        StructField("Completed", IntegerType(), nullable=True),
        StructField("On-Hold", IntegerType(), nullable=True),
        StructField("Dropped", IntegerType(), nullable=True),
        StructField("Plan to Watch", IntegerType(), nullable=True),
        StructField("Score-10", DoubleType(), nullable=True),
        StructField("Score-9", DoubleType(), nullable=True),
        StructField("Score-8", DoubleType(), nullable=True),
        StructField("Score-7", DoubleType(), nullable=True),
        StructField("Score-6", DoubleType(), nullable=True),
        StructField("Score-5", DoubleType(), nullable=True),
        StructField("Score-4", DoubleType(), nullable=True),
        StructField("Score-3", DoubleType(), nullable=True),
        StructField("Score-2", DoubleType(), nullable=True),
        StructField("Score-1", DoubleType(), nullable=True)
    ])

    df_anime = spark.read.csv(
        path=ANIME_ROUTE,
        header=True,
        sep=',',
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='Unknown',
        encoding='UTF-8',
        schema=schema
    )

    numericas, categoricas = basic_eda(df_anime, "Anime")

    plotting(df_anime.toPandas(), numericas, categoricas)

def eda_rating_complete():

    rating_schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("anime_id", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ])

    df_rating_complete = spark.read.csv(
        path=RATING_COMPLETE_ROUTE,
        header=True,
        sep=',',
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='Unknown',
        encoding='UTF-8',
        schema=rating_schema
    )

    numericas, categoricas = basic_eda(df_rating_complete, "Rating")
    sample_df = df_rating_complete.sample(fraction=0.8).toPandas()

    plotting(sample_df, numericas, categoricas)

def plotting(df_pandas, numericas, categoricas):
    sns.set_style("whitegrid")

    if numericas:
        # Histograms
        df_pandas[numericas].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Histograms of Numeric Columns", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # KDE Plots
        for num_col in numericas:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df_pandas, x=num_col, shade=True)
            plt.title(f"KDE for {num_col}")
            plt.tight_layout()
            plt.show()

        # Boxplots for Numeric Columns
        plt.figure(figsize=(15,8))
        sns.boxplot(data=df_pandas[numericas], orient='h')
        plt.title("Boxplots of Numeric Columns")
        plt.tight_layout()
        plt.show()

    # 2. Count Plots for Categorical Columns
    for cat_col in categoricas:
        plt.figure(figsize=(10, 6))
        top_values = df_pandas[cat_col].value_counts().head(20)
        sns.barplot(x=top_values.index, y=top_values.values)
        plt.title(f"Top 20 Categories of {cat_col}", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    if len(numericas) > 1:
        sns.pairplot(df_pandas[numericas], diag_kind='kde')
        plt.suptitle("Pairplot of Numeric Columns", y=1.02, fontsize=16)
        plt.show()

    if numericas:
        corr = df_pandas[numericas].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap for Numeric Columns", fontsize=16)
        plt.tight_layout()
        plt.show()

def basic_eda(df, dataframe_name):

    def get_nulls(df):
        df2 = df.select([count(when(col(c).contains('None') |
                                    col(c).contains('NULL') |
                                    (col(c) == '') |
                                    col(c).isNull() |
                                    col(c).contains('Unknown') |
                                    isnan(c), c
                                    )).alias(c)
                         for c in df.columns])
        df2.show()

        df2_pandas = df2.toPandas()
        num_rows = df.count()

        df_null_counts = df2_pandas.transpose().reset_index()
        df_null_counts.columns = ['Column', 'Null_Count']
        df_null_counts['Missing_Percentage'] = (df_null_counts['Null_Count'] / num_rows) * 100
        df_null_counts_sorted = df_null_counts.sort_values(by='Missing_Percentage', ascending=False)

        print(df_null_counts_sorted)

    print(f"Basic Exploratory Data Analysis for {dataframe_name}")
    df.show(5, truncate=False)
    df.printSchema()
    print(f"Rows: {df.count()} Columns: {len(df.columns)}")
    print(f"Data: {df.count() * len(df.columns)}")

    get_nulls(df)

    columns_types = df.dtypes
    numericas = []
    categoricas = []

    numeric_types = ('int', 'bigint', 'double', 'float', 'decimal', 'long', 'short', 'byte')
    categorical_types = ('string', 'boolean', 'date', 'timestamp', 'binary')

    for column, dtype in columns_types:
        if dtype.lower() in numeric_types:
            numericas.append(column)
        elif dtype.lower() in categorical_types:
            categoricas.append(column)
        else:
            print(f"Tipo de dato no categorizado para la columna: {column} ({dtype})")

    print("Columnas Numéricas:", numericas)
    print("Columnas Categóricas:", categoricas)

    df.select(numericas).summary().show(truncate=False)
    df.select(categoricas).show(5)
    return numericas, categoricas

if __name__ == '__main__':
    eda_df_anime()
    eda_rating_complete()
    spark.stop()