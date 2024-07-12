import pandas as pd
import psycopg2


class PostgreSQLClient:
    def __init__(
        self,
        database,
        user,
        password,
        host="localhost",
        port=5432,
    ) -> None:
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def execute_query(self, query: str) -> None:
        conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )
        cursor = conn.cursor()
        print(f"Executing query {query}")
        cursor.execute(query)
        conn.commit()
        conn.close()

    def query_to_df(self, query: str) -> pd.DataFrame:
        conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )
        cursor = conn.cursor()
        print(f"Executing query {query}")
        cursor.execute(query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        conn.commit()
        conn.close()
        return pd.DataFrame(result, columns=column_names)
