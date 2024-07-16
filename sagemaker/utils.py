import json
import math
import re

import gensim
import gensim.corpora as corpora
import nltk
from gensim.utils import simple_preprocess
from wordcloud import WordCloud

nltk.download("stopwords")
import pickle

import numpy as np
import pandas as pd
import psycopg2
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from surprise import SVD, Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate as surprise_cross_validate
from surprise.model_selection import train_test_split


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


class DataDescriptor:
    def __init__(self, data_query: str) -> None:
        with open("settings.json") as f:
            settings = json.load(f)
        postgres_settings = settings["postgresql"]
        self.__postgres = PostgreSQLClient(
            database=postgres_settings["database"],
            user=postgres_settings["user"],
            password=postgres_settings["password"],
            host=postgres_settings["host"],
            port=postgres_settings["port"],
        )
        self.data = self.__postgres.query_to_df(data_query)

    def get_histograms(self, variables: list, bins_amount: dict = None) -> None:
        """
        Function that creates a plot with histograms as subplots
        according to data, variables and bins amount parameters.
        """
        df = self.data[variables]

        if bins_amount is None:
            bins_amount = {}
            for var in variables:
                bins_amount[var] = 10

        plt.figure(figsize=(18, 20))
        for n, col in enumerate(df.columns, 1):
            plt.subplot(4, 5, n)
            sns.histplot(df[col], bins=bins_amount[col])
            plt.title(f"Distr {col}")
        plt.show()

    def get_barplots(self, variables: list, charts_by_col: int = 3) -> None:
        """
        Function that creates a plot with barplots as subplots
        according to data and variables parameters.
        """
        df = self.data[variables]
        plt.figure(figsize=(15, 5))
        row_num = math.ceil(len(variables) / charts_by_col)
        for n, col in enumerate(df.columns, 1):
            plt.subplot(row_num, charts_by_col, n)
            count_df = (
                pd.DataFrame(round(df[col].value_counts() / len(df[col]) * 100, 2))
                .rename(columns={col: "count"})
                .reset_index(names=col)
            )
            bars = plt.bar(count_df[col], count_df["count"])
            plt.ylabel("Pct of total")
            plt.xlabel(col)
            plt.bar_label(bars)
        plt.show()

    def get_boxplots(self, target_variable: str, split_variables: str = None) -> None:
        """
        Function that creates a plot with boxplots as subplots
        according to data and variables parameters.
        """
        df = self.data
        plt.figure(figsize=(15, 5))
        if split_variables is None:
            ax = sns.boxplot(y=target_variable, data=df)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.title(f"{target_variable} boxplot")
            plt.ylabel(f"{target_variable} values")
        else:
            ax = sns.boxplot(x=split_variables, y=target_variable, data=df)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.title(f"{target_variable} boxplots by {split_variables}")
            plt.ylabel(f"{target_variable} values")
            plt.xlabel(split_variables)
        plt.show()

    def get_scatterplots_features_vs_target(
        self, target_variable: str, variables: list
    ) -> None:
        """
        Function that creates a plot with scatterplots as subplots
        according to data, feature and target variables parameters.
        """
        df = self.data
        feature_df = df[variables]
        target_df = df[target_variable]
        fig_tot = len(feature_df.columns)
        fig_por_fila = 4
        tamanio_fig = 4
        num_filas = int(np.ceil(fig_tot / fig_por_fila))
        plt.figure(
            figsize=(fig_por_fila * tamanio_fig + 5, num_filas * tamanio_fig + 5)
        )
        for i, col in enumerate(feature_df.columns, 1):
            plt.subplot(num_filas, fig_por_fila, i)
            sns.scatterplot(x=feature_df[col], y=target_df)
            plt.title("%s vs %s" % (col, target_variable))
            plt.ylabel("Revenue")
            plt.xlabel(col)
        plt.show()

    def get_corr_matrix(self, variables: list) -> pd.DataFrame:
        """
        Function that creates a plot of the correlation matrix as a
        heatmap according to data and variables parameters.
        """
        df = self.data[variables]
        sns.heatmap(
            df.corr(method="pearson"),
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            cmap="Blues",
        )
        plt.show()


class LDA:
    def __init__(self, texts_query: str, texts_col: str) -> None:
        with open("settings.json") as f:
            settings = json.load(f)

        postgres_settings = settings["postgresql"]
        self.__postgres = PostgreSQLClient(
            database=postgres_settings["database"],
            user=postgres_settings["user"],
            password=postgres_settings["password"],
            host=postgres_settings["host"],
            port=postgres_settings["port"],
        )

        self.data = self.__postgres.query_to_df(texts_query)
        self.texts_col = texts_col

    def get_words_cloud(self, max_words: int = 5000):
        wordcloud = WordCloud(
            background_color="white",
            max_words=max_words,
            contour_width=3,
            contour_color="steelblue",
        )
        wordcloud.generate(",".join(list(self.data[self.texts_col].values)))
        return wordcloud.to_image()

    def train_model(self, extra_stop_words: list, num_topics: int = 5) -> None:
        self.num_topics = num_topics
        stop_words = stopwords.words("english")
        stop_words.extend(extra_stop_words)

        self.__texts = self.data[self.texts_col].values.tolist()
        self.__texts = [s.lower() for s in self.__texts]
        self.__texts = [re.sub("[,\.!?]", "", s) for s in self.__texts]
        self.__texts = [
            gensim.utils.simple_preprocess(str(s), deacc=True) for s in self.__texts
        ]
        self.__texts = [
            [word for word in simple_preprocess(str(t)) if word not in stop_words]
            for t in self.__texts
        ]
        self.__id2word = corpora.Dictionary(self.__texts)
        self.__corpus = [self.__id2word.doc2bow(s) for s in self.__texts]
        self.__lda_model = gensim.models.LdaMulticore(
            corpus=self.__corpus, id2word=self.__id2word, num_topics=self.num_topics
        )

    def get_topics_graphics(self) -> None:
        pyLDAvis.enable_notebook()
        LDA_results_filepath = (
            f"./results/lda_{self.texts_col}_{self.num_topics}_topics"
        )
        LDA_prepared = pyLDAvis.gensim.prepare(
            self.__lda_model, self.__corpus, self.__id2word
        )
        with open(LDA_results_filepath, "wb") as f:
            pickle.dump(LDA_prepared, f)
        with open(LDA_results_filepath, "rb") as f:
            LDA_prepared = pickle.load(f)
        pyLDAvis.save_html(
            LDA_prepared, f"./results/lda_{self.texts_col}_{self.num_topics}.html"
        )

    def predict_topics(self) -> pd.DataFrame:
        self.topics_predictions = self.data
        self.topics_predictions["topics_distr"] = self.topics_predictions[
            self.texts_col
        ].apply(lambda x: self.__lda_model[self.__id2word.doc2bow(x.lower().split())])

        items_list = []
        for item in self.topics_predictions.to_dict("records"):
            for topic_distr in item["topics_distr"]:
                item[f"{self.texts_col}_topic_{topic_distr[0]+1}"] = topic_distr[1]
            items_list.append(item)

        self.topics_predictions = (
            pd.DataFrame(items_list).drop(columns=["topics_distr"]).fillna(0)
        )
        return self.topics_predictions

    def upload_topics(self) -> None:
        create_table_query = f"""
            DROP TABLE IF EXISTS animes_catalog_{self.texts_col}_topics;

            CREATE TABLE animes_catalog_{self.texts_col}_topics (
                {self.texts_col}_topics_id INT GENERATED ALWAYS AS IDENTITY,
                anime_id INT NOT NULL,

                primary key({self.texts_col}_topics_id),
                foreign key(anime_id)
                references animes_catalog(anime_id)
            );
        """
        self.__postgres.execute_query(create_table_query)

        for t in range(self.num_topics):
            add_column_query = f"""
                ALTER TABLE animes_catalog_{self.texts_col}_topics
                ADD {self.texts_col}_topic_{t+1} DECIMAL(16, 4) NOT NULL;
            """
            self.__postgres.execute_query(add_column_query)

        anime_catalog_topics_insert_data = ", ".join(
            [
                f"""({record["anime_id"]}, {", ".join([str(record[f"{self.texts_col}_topic_{t + 1}"]) for t in range(self.num_topics)])})"""
                for record in self.topics_predictions.to_dict("records")
            ]
        )
        if anime_catalog_topics_insert_data:
            self.__postgres.execute_query(
                query=f"""
                    INSERT INTO animes_catalog_{self.texts_col}_topics (anime_id, {", ".join([f"{self.texts_col}_topic_{t + 1}" for t in range(self.num_topics)])})
                    VALUES {anime_catalog_topics_insert_data};
                """
            )


class Clustering:
    def __init__(self, data_query: str, features_replace_values: dict = None) -> None:
        with open("settings.json") as f:
            settings = json.load(f)
        postgres_settings = settings["postgresql"]
        self.__postgres = PostgreSQLClient(
            database=postgres_settings["database"],
            user=postgres_settings["user"],
            password=postgres_settings["password"],
            host=postgres_settings["host"],
            port=postgres_settings["port"],
        )
        self.data = self.__postgres.query_to_df(data_query)
        self.replace_values = features_replace_values

    def get_elbow_plot(
        self, k_options: list, model_features_list: list = None, cv: int = 3
    ) -> None:
        df = self.data
        if model_features_list is not None:
            df = df[model_features_list]

        if self.replace_values is not None:
            for col in self.replace_values.keys():
                df[col] = df[col].replace(self.replace_values[col])

        self.__data_array = np.array(df)

        kmeans = [
            KMeans(init="random", random_state=0, n_clusters=k) for k in k_options
        ]
        results = [
            sklearn_cross_validate(kmeans[k], self.__data_array, cv=cv)
            for k in range(len(k_options))
        ]
        scores = [np.mean(results[k]["test_score"]) * -1 for k in range(len(k_options))]
        plt.plot(k_options, scores)
        plt.xlabel("Param k")
        plt.ylabel("Score")
        plt.title("Score by K")
        plt.show()

    def train_model(self, k: int) -> None:
        self.__clustering_model = KMeans(init="random", random_state=0, n_clusters=k)
        self.__clustering_model.fit(self.__data_array)

    def predict_clusters(self):
        self.clustering_predictions = self.__clustering_model.predict(self.__data_array)
        self.clustering_predictions = pd.DataFrame(self.clustering_predictions).rename(
            columns={0: "cluster"}
        )
        self.clustering_predictions["cluster"] = (
            self.clustering_predictions["cluster"] + 1
        )
        self.clustering_predictions = pd.merge(
            self.data, self.clustering_predictions, left_index=True, right_index=True
        )
        return self.clustering_predictions

    def __get_clusters_distr(self) -> None:
        if "user_id" in self.clustering_predictions.columns:
            data = self.clustering_predictions[["cluster", "user_id"]]
        elif "anime_id" in self.clustering_predictions.columns:
            data = self.clustering_predictions[["cluster", "anime_id"]]
        (
            data.groupby(["cluster"]).count() / len(self.clustering_predictions)
        ).plot.bar()
        plt.show()

    def __get_qualitative_vars_distr_by_cluster(
        self, qualitative_var: str, charts_by_row
    ):
        data = self.clustering_predictions
        plt.figure(figsize=(15, 15))
        clusters_amount = len(data["cluster"].unique())
        row_num = math.ceil(clusters_amount / charts_by_row)
        for k in range(1, clusters_amount + 1):
            data_temp_k = data[data["cluster"] == k]
            plt.subplot(row_num, charts_by_row, k)
            bars = plt.bar(
                data_temp_k[qualitative_var].unique(),
                data_temp_k[qualitative_var].value_counts() / len(data_temp_k),
            )
            plt.ylabel("Pct of total")
            plt.xlabel(qualitative_var)
            plt.title(f"Distribution by {qualitative_var} values - cluster {k}")
        plt.show()

    def __get_quantitative_var_boxplot_by_cluster(self, quantitative_var: str):
        data = self.clustering_predictions[["cluster", quantitative_var]]
        ax = sns.boxplot(x="cluster", y=quantitative_var, data=data)
        plt.title(f"{quantitative_var} boxplots by cluster")
        plt.ylabel(f"{quantitative_var} values")
        plt.xlabel("cluster")
        plt.show()

    def get_cluster_graphics(
        self,
        qualitative_vars: list[str] = None,
        quantitative_vars: list[str] = None,
        charts_by_row: int = 2,
    ) -> None:
        self.__get_clusters_distr()
        if qualitative_vars is not None:
            for var in qualitative_vars:
                self.__get_qualitative_vars_distr_by_cluster(
                    qualitative_var=var, charts_by_row=charts_by_row
                )
        if quantitative_vars is not None:
            for var in quantitative_vars:
                self.__get_quantitative_var_boxplot_by_cluster(quantitative_var=var)


class RecommendationSystem:
    def __init__(
        self,
        rating_data_query: str,
        user_id_col: str,
        item_id_col: str,
        rating_col: str,
        min_rating: int,
        max_rating: int,
    ) -> None:
        with open("settings.json") as f:
            settings = json.load(f)
        postgres_settings = settings["postgresql"]
        self.__postgres = PostgreSQLClient(
            database=postgres_settings["database"],
            user=postgres_settings["user"],
            password=postgres_settings["password"],
            host=postgres_settings["host"],
            port=postgres_settings["port"],
        )
        self.rating_data = self.__postgres.query_to_df(rating_data_query)
        self.__rating_dataset = Dataset.load_from_df(
            self.rating_data[[user_id_col, item_id_col, rating_col]],
            Reader(rating_scale=(min_rating, max_rating)),
        )

    def evaluate_params(
        self,
        model: str,
        model_param_grid: dict,
        number_cross_validations: int = 5,
    ) -> GridSearchCV:
        if model.lower() == "svd":
            gridsearch_cv = GridSearchCV(
                algo_class=SVD,
                param_grid=model_param_grid,
                measures=["rmse"],
                cv=number_cross_validations,
            )
        elif model.lower() == "knn":
            gridsearch_cv = GridSearchCV(
                algo_class=KNNWithMeans,
                param_grid=model_param_grid,
                measures=["rmse"],
                cv=number_cross_validations,
            )
        gridsearch_cv.fit(self.__rating_dataset)
        return gridsearch_cv

    def evaluate_execution_time(
        self, model: str, model_params: dict, number_cross_validations: int = 5
    ) -> dict:
        if model.lower() == "svd":
            algorythm = SVD(**model_params)
        elif model.lower() == "knn":
            algorythm = KNNWithMeans(**model_params)
        algo_results = surprise_cross_validate(
            algo=algorythm,
            data=self.__rating_dataset,
            measures=["rmse"],
            cv=number_cross_validations,
            verbose=False,
        )
        return algo_results

    def train_model(
        self, model: str, model_params: dict, test_size_pct: float = 0.2
    ) -> None:
        trainset, testset = train_test_split(
            self.__rating_dataset, test_size=test_size_pct
        )
        if model.lower() == "svd":
            self.__algorythm = SVD(**model_params)
        elif model.lower() == "knn":
            self.__algorythm = KNNWithMeans(**model_params)
        self.__algorythm.fit(trainset)
        predictions = self.__algorythm.test(testset)
        print(accuracy.rmse(predictions))

    def predict_recommendations(
        self,
        items_query: str,
        items_id_col: str,
        items_name_col: str,
        user_id: int,
        num_recos: int = 10,
        threshold: int = 6,
    ) -> pd.DataFrame:
        predictions = []
        items = self.__postgres.query_to_df(query=items_query)
        items = items.to_dict("records")
        for item in items:
            prediction = self.__algorythm.predict(uid=user_id, iid=item[items_id_col])
            if (
                prediction.details["was_impossible"] == False
                and prediction.est >= threshold
            ):
                predictions.append(
                    {
                        "user_id": user_id,
                        "anime_id": item[items_id_col],
                        "anime_name": item[items_name_col],
                        "rating": round(prediction.est, 2),
                    }
                )
        self.recommendations_predictions = pd.DataFrame(predictions)
        self.recommendations_predictions = (
            self.recommendations_predictions.sort_values(by=["rating"], ascending=False)
            .head(num_recos)
            .reset_index(drop=True)
        )
        return self.recommendations_predictions
