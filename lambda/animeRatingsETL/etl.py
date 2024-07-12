import json
from io import StringIO

import boto3
import pandas as pd
from utils import PostgreSQLClient


class AnimeRatingsETL:
    def __init__(self, s3_bucket: str, s3_key: str) -> None:
        with open("settings.json") as f:
            settings = json.load(f)
        postgres_settings = settings["postgresql"]
        self.postgres = PostgreSQLClient(
            database=postgres_settings["database"],
            user=postgres_settings["user"],
            password=postgres_settings["password"],
            host=postgres_settings["host"],
            port=postgres_settings["port"],
        )
        self.s3_client = boto3.client("s3")
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        print(
            f"AnimeRatingsETL initialized with params: s3_bucket: {self.s3_bucket}, s3_key: {self.s3_key}"
        )

    def read_raw_data(self) -> None:
        print("Executing read_raw_data method")
        anime_ratings_object = self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=self.s3_key,
        )
        anime_ratings_csv = anime_ratings_object["Body"].read().decode("utf-8")
        self.raw_data = {
            "raw_anime_ratings": pd.read_csv(StringIO(anime_ratings_csv)),
        }

    def prepare_df(self) -> None:
        db_anime_ids = self.postgres.query_to_df(
            query="""
                SELECT DISTINCT anime_id
                FROM animes_catalog;
            """
        )["anime_id"].values.tolist()
        anime_ratings = self.raw_data["raw_anime_ratings"]
        anime_ratings = anime_ratings[anime_ratings["rating"] > 0]
        anime_ratings = anime_ratings[
            anime_ratings["anime_id"].apply(lambda x: x in db_anime_ids)
        ]
        self.processed_data = {"anime_ratings": anime_ratings}

    def upload_data(self) -> None:
        anime_ratings = self.processed_data["anime_ratings"].to_dict("records")
        anime_ratings_insert_data = ", ".join(
            [
                f"""({rating["user_id"]}, '{rating["anime_id"]}', '{rating["rating"]}')"""
                for rating in anime_ratings
            ]
        )
        if anime_ratings_insert_data:
            self.postgres.execute_query(
                query=f"""
                    INSERT INTO animes_ratings (user_id, anime_id, rating)
                    VALUES {anime_ratings_insert_data};
                """
            )
