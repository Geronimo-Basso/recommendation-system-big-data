import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, StringType
)

ROOT = 'data/'
ANIME_ROUTE = ROOT + 'anime.csv'
RATING_COMPLETE_ROUTE = ROOT + 'rating_complete.csv'
RATING_EP_ROUTE = ROOT + 'valoraciones_EP.csv'
cpu_cores = multiprocessing.cpu_count()
spark = SparkSession.builder.master(f"local[{cpu_cores - 1}]").appName("Reviews").getOrCreate()

def upload_data():
    schema_anime = StructType([
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
        schema=schema_anime
    )

    schema_rating = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("anime_id", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ])

    df_ratings = spark.read.csv(
        path=RATING_COMPLETE_ROUTE,
        header=True,
        sep=',',
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='Unknown',
        encoding='UTF-8',
        schema=schema_rating
    )

    schema_rating_ep = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("anime_id", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ])

    df_rating_ep = spark.read.csv(
        path=RATING_EP_ROUTE,
        header=False,
        sep=',',
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='Unknown',
        encoding='UTF-8',
        schema=schema_rating_ep
    )

    print(df_anime.count())
    print(df_ratings.count())
    print(df_rating_ep.count())

    return df_anime, df_ratings, df_rating_ep


def join_dataset(anime_dataframe, ratings_dataframe, ratings_ep_dataframe):
    joined_ratings_dataframe = ratings_dataframe.join(anime_dataframe, ratings_dataframe.anime_id == anime_dataframe.ID,
                                                      'inner')
    joined_rating_ep_dataframe = ratings_ep_dataframe.join(anime_dataframe,
                                                           ratings_ep_dataframe.anime_id == anime_dataframe.ID, 'inner')

    joined_ratings_dataframe.show(truncate=False)
    joined_rating_ep_dataframe.show(truncate=False)

    return joined_ratings_dataframe, joined_rating_ep_dataframe

def preprocess_data(dataframe_to_be_preprocessed):
    columns_to_drop = ['Licensors','Premiered','English name','Japanese name', 'Producers', 'Studios']
    dataframe_to_be_preprocessed = spark.createDataFrame(dataframe_to_be_preprocessed, columns_to_drop)
    return dataframe_to_be_preprocessed


if __name__ == '__main__':
    df_anime, df_ratings, df_rating_ep = upload_data()
    joined_ratings, joined_rating_ep = join_dataset(df_anime, df_ratings, df_rating_ep)

    spark.stop()
