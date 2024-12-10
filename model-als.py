import multiprocessing
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, StringType
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

ROOT = 'data/'
ANIME_ROUTE = ROOT + 'anime.csv'
RATING_COMPLETE_ROUTE = ROOT + 'rating_complete.csv'
RATING_EP_ROUTE = ROOT + 'valoraciones_EP.csv'


def load_anime_data(spark):
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
        quote='"',
        escape='"',
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
        nullValue='Unknown',
        encoding='UTF-8',
        schema=schema
    )

    return df_anime


def load_ratings(spark):
    rating_schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("anime_id", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ])

    df_rating_complete = spark.read.csv(
        path=RATING_COMPLETE_ROUTE,
        header=True,
        sep=',',
        nullValue='Unknown',
        encoding='UTF-8',
        schema=rating_schema
    )

    ep_schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("anime_id", IntegerType(), True),
        StructField("rating", DoubleType(), True)
    ])

    df_valoraciones_ep = spark.read.csv(
        path=RATING_EP_ROUTE,
        header=False,
        sep=',',
        nullValue='Unknown',
        encoding='UTF-8',
        schema=ep_schema
    )

    return df_rating_complete, df_valoraciones_ep


def train_recommender(df_ratings, df_valoraciones_ep, df_anime, user_id=666666):
    combined_ratings = df_ratings.union(df_valoraciones_ep).dropna()

    combined_ratings = combined_ratings.dropDuplicates(['user_id', 'anime_id'])

    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="user_id",
        itemCol="anime_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=10
    )

    (train, test) = combined_ratings.randomSplit([0.8, 0.2], seed=42)

    model = als.fit(train)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error (RMSE) = {rmse}")

    target_user_df = spark.createDataFrame([(user_id,)], ["user_id"])
    user_recs = model.recommendForUserSubset(target_user_df, 30)

    ep_recs = user_recs.filter(col("user_id") == user_id).select("recommendations").collect()

    recs = ep_recs[0]["recommendations"]
    recs_list = [(row["anime_id"], row["rating"]) for row in recs]

    recommendations_df = spark.createDataFrame(recs_list, ["anime_id", "predicted_rating"])

    recs_with_info = recommendations_df.join(df_anime, recommendations_df.anime_id == df_anime.ID, how="left") \
        .select(df_anime.ID, df_anime.Name, df_anime["English name"],
                df_anime.Type, recommendations_df .predicted_rating)

    avg_ratings = df_ratings.groupBy("anime_id").avg("rating").withColumnRenamed("avg(rating)", "avg_rating")

    recs_with_avg = recs_with_info.join(avg_ratings, recs_with_info.ID == avg_ratings.anime_id, how="left") \
        .drop(avg_ratings.anime_id)

    tv_recs = recs_with_avg.filter(col("Type") == "TV") \
        .orderBy(col("avg_rating").desc()) \
        .select("ID", "Name", "English name", "avg_rating") \
        .limit(5)

    movie_recs = recs_with_avg.filter(col("Type") == "Movie") \
        .orderBy(col("avg_rating").desc()) \
        .select("ID", "Name", "English name", "avg_rating") \
        .limit(5)

    return tv_recs, movie_recs


def save_recommendations(tv_recs, movie_recs):
    tv_recs_pd = tv_recs.toPandas()
    movie_recs_pd = movie_recs.toPandas()

    tv_recs_sorted = tv_recs_pd.sort_values(by="avg_rating", ascending=False)
    movie_recs_sorted = movie_recs_pd.sort_values(by="avg_rating", ascending=False)

    tv_recs_final = tv_recs_sorted[["ID", "Name", "English name", "avg_rating"]]
    movie_recs_final = movie_recs_sorted[["ID", "Name", "English name", "avg_rating"]]

    tv_recs_final.to_csv("recommendations_series.csv", index=False, sep=',')
    movie_recs_final.to_csv("recommendations_movies.csv", index=False, sep=',')


if __name__ == '__main__':
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("org").setLevel(logging.ERROR)

    cpu_cores = multiprocessing.cpu_count()
    spark = (
        SparkSession.builder.master(f"local[{cpu_cores - 1}]").
        appName("RecommenderSystem").
        config("spark.driver.memory", "4g").
        config("spark.executor.memory", "4g").
        config("spark.memory.offHeap.enabled", "true").
        config("spark.memory.offHeap.size", "2g").
        getOrCreate()
    )

    df_anime = load_anime_data(spark)
    df_rating_complete, df_valoraciones_ep = load_ratings(spark)

    print("Training ALS model and generating recommendations...")
    tv_recs, movie_recs = train_recommender(df_rating_complete, df_valoraciones_ep, df_anime)
    save_recommendations(tv_recs, movie_recs)
    spark.stop()
