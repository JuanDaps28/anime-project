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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
from nltk.corpus import stopwords
from surprise import SVD, Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import (GridSearchCV, cross_validate,
                                      train_test_split)


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

    def add_data_topics(self) -> pd.DataFrame:
        self.data["topics_distr"] = self.data[self.texts_col].apply(
            lambda x: self.__lda_model[self.__id2word.doc2bow(x.lower().split())]
        )

        items_list = []
        for item in self.data.to_dict("records"):
            for topic_distr in item["topics_distr"]:
                item[f"topic_{topic_distr[0]+1}"] = topic_distr[1]
            items_list.append(item)

        self.data = pd.DataFrame(items_list).drop(columns=["topics_distr"]).fillna(0)
        return self.data


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
        algo_results = cross_validate(
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
        predictions = pd.DataFrame(predictions)
        predictions = (
            predictions.sort_values(by=["rating"], ascending=False)
            .head(num_recos)
            .reset_index(drop=True)
        )
        return predictions
