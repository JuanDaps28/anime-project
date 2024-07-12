from etl import AnimeCatalogETL


def lambda_handler(event, context):
    anime_catalog_etl = AnimeCatalogETL(
        s3_bucket=event["Records"][0]["s3"]["bucket"]["name"],
        s3_key=event["Records"][0]["s3"]["object"]["key"],
    )
    anime_catalog_etl.read_raw_data()
    anime_catalog_etl.prepare_df()
    anime_catalog_etl.upload_data()
    anime_catalog_etl.postgres.close_connection()
