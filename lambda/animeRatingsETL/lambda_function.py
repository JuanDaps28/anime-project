from etl import AnimeRatingsETL


def lambda_handler(event, context):
    anime_ratings_etl = AnimeRatingsETL(
        s3_bucket=event["Records"][0]["s3"]["bucket"]["name"],
        s3_key=event["Records"][0]["s3"]["object"]["key"],
    )
    anime_ratings_etl.read_raw_data()
    anime_ratings_etl.prepare_df()
    anime_ratings_etl.upload_data()
    anime_ratings_etl.postgres.close_connection()
