import io

from src.preprocess import DataPreProcessor
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DBScan:
    """
    The class performs DBSCAN clustering of the dataset, but first it performs
    a cross validation using GridSearchCV, to find the best eps and min_samples.
    It provides 3 methods:
    - search_hyperparameters(): To find the best hyperparameters for DBSCAN, as
    well as best possible validation score and labels.
    - cluster(): Actually form the clusters.
    - visualize(): Visualize the clusters in the dataset using colormaps.
    """

    def __init__(self, dataframe: dd.DataFrame, target_index: int) -> None:
        """
        Preprocesses the data for training, testing and validation.
        :param dataframe: The dataframe to cluster.
        :param target_index: The index of the target column.
        """
        preprocess = DataPreProcessor(dataframe, target_index)
        train, test, validation = preprocess.preprocess()

        self.train = train.compute()
        self.test = test.compute()
        self.validation = validation.compute()

        self.model = None
        self.predictions = None

    def search_hyperparameters(self) -> list:
        """
        The function finds the best hyperparameters, labels and
        silhouette score for the model.
        :return: A list of hyperparameters, score and labels.
        """
        best_score = -1
        best_params = None
        best_labels = None

        params_grid = {
            "eps": np.arange(0.1, 1.1, 0.1),
            "min_samples": range(2, 10)
        }

        self.validation.columns = self.validation.columns.astype(str)

        for params in ParameterGrid(params_grid):
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels = dbscan.fit_predict(self.validation)

            if len(set(labels)) > 1:
                score = silhouette_score(self.validation, labels)
                if score > best_score:
                    best_params = params
                    best_labels = labels
                    best_score = score

        return [best_params, best_score, best_labels]

    def cluster(self) -> None:
        """
        The function finds the best hyperparameters and initializes a model
        for training, which finds clusters.
        :return: None
        """

        test_results = self.search_hyperparameters()
        self.train.columns = self.train.columns.astype(str)

        params = test_results[0]
        self.model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        self.predictions = self.model.fit_predict(self.train)

        self.train["clusters"] = self.predictions

    def visualize(self) -> io.BytesIO:
        """
        The function visualizes the clusters using a colormap.
        :return: A BytesIO object of the image.
        """
        plt.figure(figsize=(10, 12))
        sns.scatterplot(x=self.train.iloc[:, 0], y=self.train.iloc[:, 1], hue=self.train["clusters"], palette="viridis")
        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Convert plot to bytes stream
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf

    def score(self) -> float:
        """
        The function finds and returns the silhouette score
        of the clustering.
        :return: The silhouette score.
        """
        score = silhouette_score(self.train, self.predictions)
        return score


def main():
    """
    Test function. Run the file to view the functionality.
    :return: None.
    """
    read = dd.read_csv("../data/weather.csv")

    dbscan_test = DBScan(read, 10)
    dbscan_test.cluster()
    dbscan_test.visualize()


if __name__ == "__main__":
    main()
