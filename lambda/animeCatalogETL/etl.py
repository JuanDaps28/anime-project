import json
from io import StringIO

import boto3
import numpy as np
import pandas as pd
from utils import PostgreSQLClient


class AnimeCatalogETL:
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
            f"AnimeCatalogETL initialized with params: s3_bucket: {self.s3_bucket}, s3_key: {self.s3_key}"
        )

    def read_raw_data(self) -> None:
        print("Executing read_raw_data method")
        anime_synopsis_object = self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=self.s3_key,
        )
        anime_synopsis_csv = anime_synopsis_object["Body"].read().decode("utf-8")
        anime_object = self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=self.s3_key.replace("anime_with_synopsis", "anime"),
        )
        anime_csv = anime_object["Body"].read().decode("utf-8")
        self.raw_data = {
            "raw_anime_catalog": pd.read_csv(StringIO(anime_csv)),
            "raw_anime_synopsis": pd.read_csv(StringIO(anime_synopsis_csv)),
        }

    def prepare_df(self) -> None:
        print("Executing prepare_df method")
        raw_anime_catalog = self.raw_data["raw_anime_catalog"][
            ["anime_id", "name", "genre", "type", "episodes"]
        ].set_index("anime_id")

        raw_anime_synopsis = (
            self.raw_data["raw_anime_synopsis"][["MAL_ID", "Genres", "sypnopsis"]]
            .rename(
                columns={
                    "MAL_ID": "anime_id",
                    "Genres": "genre",
                    "sypnopsis": "synopsis",
                }
            )
            .set_index("anime_id")
        )

        print("Joining anime_synopsis df into anime df")
        anime_catalog = raw_anime_catalog.join(
            raw_anime_synopsis, how="left", rsuffix="_synopsis"
        ).reset_index()
        anime_catalog["genre"] = anime_catalog.apply(
            lambda x: x["genre_synopsis"] if pd.isna(x["genre"]) else x["genre"], axis=1
        )
        anime_catalog.drop(columns=["genre_synopsis"], inplace=True)

        for col in ["genre", "type", "synopsis"]:
            print(f"Replacing NA values with empty string in {col} column")
            anime_catalog[col].fillna("", inplace=True)

        print("Replacing No synopsis with empty string in synopsis column")
        anime_catalog["synopsis"] = anime_catalog["synopsis"].apply(
            lambda x: "" if "No synopsis" in x else x
        )

        print("Replacing Unknown values with NA in episodes column")
        anime_catalog["episodes"].replace(
            to_replace="Unknown", value=np.nan, inplace=True
        )
        print(f"Replacing Unknown values with empty string in genre column")
        anime_catalog["genre"].replace(to_replace="Unknown", value="", inplace=True)

        replace_values = {
            "&amp;": "&",
            "&gt": ">",
            "&#039;": "",
            "&quot;": '"',
            "'": "",
        }

        for col in ["name", "genre", "synopsis"]:
            for k, v in replace_values.items():
                print(f"Replacing {k} for {v} in column {col}")
                anime_catalog[col] = anime_catalog[col].apply(lambda x: x.replace(k, v))

        anime_catalog = anime_catalog.astype(
            {
                "anime_id": "int64",
                "name": "object",
                "genre": "object",
                "type": "object",
                "episodes": "float64",
                "synopsis": "object",
            }
        )
        self.processed_data = {"anime_catalog": anime_catalog}

    def upload_data(self) -> None:
        print("Executing upload_data method")
        self.__update_deleted_animes()
        self.__update_existing_animes()
        self.__insert_new_animes()

    def __update_deleted_animes(self) -> None:
        anime_catalog_ids = self.processed_data["anime_catalog"][
            "anime_id"
        ].values.tolist()
        anime_catalog_ids = [str(anime) for anime in anime_catalog_ids]
        self.postgres.execute_query(
            query=f"""
                UPDATE animes_catalog
                SET deleted = True
                WHERE anime_id IN (
                    SELECT DISTINCT anime_id
                    FROM animes_catalog
                    WHERE anime_id not in ({", ".join(anime_catalog_ids)})
                );
            """
        )

    def __update_existing_animes(self) -> None:
        db_anime_ids = self.postgres.query_to_df(
            query="""
                SELECT DISTINCT anime_id
                FROM animes_catalog;
            """
        )["anime_id"].values.tolist()
        anime_catalog = self.processed_data["anime_catalog"].to_dict("records")
        anime_catalog = [
            anime for anime in anime_catalog if anime["anime_id"] in db_anime_ids
        ]
        for anime in anime_catalog:
            self.postgres.execute_query(
                query=f"""
                    UPDATE animes_catalog
                    SET
                        name = '{anime["name"]}',
                        genre = '{anime["genre"]}',
                        type = '{anime["type"]}',
                        episodes = {"NULL" if pd.isna(anime["episodes"]) else anime["episodes"]},
                        synopsis = '{anime["synopsis"]}',
                        deleted = False
                    WHERE anime_id = {anime["anime_id"]};
                """
            )

    def __insert_new_animes(self) -> None:
        db_anime_ids = self.postgres.query_to_df(
            query="""
                SELECT DISTINCT anime_id
                FROM animes_catalog;
            """
        )
        db_anime_ids = db_anime_ids["anime_id"].values.tolist()
        anime_catalog = self.processed_data["anime_catalog"].to_dict("records")
        anime_catalog_insert_data = ", ".join(
            [
                f"""({anime["anime_id"]}, '{anime["name"]}', '{anime["genre"]}', '{anime["type"]}', {"NULL" if pd.isna(anime["episodes"]) else anime["episodes"]}, '{anime["synopsis"]}', False)"""
                for anime in anime_catalog
                if anime["anime_id"] not in db_anime_ids
            ]
        )
        if anime_catalog_insert_data:
            self.postgres.execute_query(
                query=f"""
                    INSERT INTO animes_catalog (anime_id, name, genre, type, episodes, synopsis, deleted)
                    VALUES {anime_catalog_insert_data};
                """
            )
