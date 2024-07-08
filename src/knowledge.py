import dask.dataframe as dd
import logging
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scipy.stats import norm

"""
The file contains only one class. The class provides all the data required
to gain knowledge about the dataset and particular columns.

It also returns relevant graphs and charts.
"""


def write_logs(error: bool, message: str, error_reason: Exception = None) -> None:
    """
    The function automates the process of writing to the log file.
    :param error: If there has been an error, pass True.
    :param message: The message to written to the log file.
    :param error_reason: If there is an error, pass the exception to raise it.
    :return: None.
    """
    if error:
        logging.info(msg=message)
        raise error_reason
    else:
        logging.info(msg=message)


class Knowledge:
    """
    The class performs all the tasks required to perform knowledge
    representation of the dataset on the user interface.

    The class majorly contains 4 methods
    - basic_data() - To return basic data about the dataset.
    - column_data() - To return information pertaining to a column.
    - correlation_matrix() - To return the correlation matrix.
    - feature_histogram() - To return the feature histogram of the matrix.
    """

    def __init__(self, input_dataframe: dd.DataFrame, numerical: list[str], categorical: list[str]):
        """
        The function initializes class attributes and logging.
        :param input_dataframe: The data set to perform knowledge data acquisition.
        :param numerical: The numerical columns in the dataset.
        :param categorical: The categorical columns in the dataset.
        """
        # Initializing attributes.
        self.df = input_dataframe
        self.numerical = numerical
        self.categorical = categorical

        # Initializing logging.
        logging.basicConfig(
            level=logging.INFO,
            filemode="a",
            filename="../reports/logs.log",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        write_logs(message="Constructor Executed Successfully.", error=False)

    def basic_data(self, before_preprocess: bool) -> dict:
        """
        The function finds all the basic information about the dataset.
        :param before_preprocess: Avoids the correlation matrix, as it needs encoding.
        :return: A dictionary of basic information.
        """
        try:
            if not before_preprocess:
                data_dict = dict()

                # Finding the number of rows in the dataset.
                rows = len(self.df)
                data_dict["Rows"] = rows

                # Finding the number of duplicates in the dataframe.
                pandas_df = self.df.compute()
                duplicated = rows - len(pandas_df.duplicated())
                data_dict["Duplicates"] = duplicated

                # Finding the memory usage.
                memory = pandas_df.memory_usage(deep=True).sum()
                data_dict["RAM"] = str(f"{memory} MB")

                # Finding the number of categorical features.
                categorical_features = len(self.categorical)
                data_dict["Categorical Features"] = categorical_features

                # Finding the number of numerical features.
                numerical_features = len(self.numerical)
                data_dict["Numerical Features"] = numerical_features

                # Finding the correlation matrix and storing as a BytesIO object.
                corr_matrix = pandas_df.corr()
                plt.figure(figsize=(20, 16))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    xticklabels=corr_matrix.columns,
                    yticklabels=corr_matrix.columns
                )

                corr_img = io.BytesIO()
                plt.savefig(corr_img, format="png")
                plt.close()
                corr_img.seek(0)

                data_dict["Correlation"] = corr_img

                # Finding the feature histogram and storing as a BytesIO object.
                pandas_df.hist(figsize=(20, 20))

                hist_img = io.BytesIO()
                plt.savefig(hist_img, format="png")
                plt.close()
                hist_img.seek(0)

                data_dict["Histogram"] = hist_img

                write_logs(
                    error=False,
                    message=f"Successfully Found Feature Histogram"
                )

                return data_dict

            else:
                data_dict = dict()

                # Finding the number of rows in the dataset.
                rows = len(self.df)
                data_dict["Rows"] = rows

                # Finding the number of duplicates in the dataframe.
                pandas_df = self.df.compute()
                duplicated = rows - len(pandas_df.duplicated())
                data_dict["Duplicates"] = duplicated

                # Finding the memory usage.
                memory = pandas_df.memory_usage(deep=True).sum()
                data_dict["RAM"] = str(f"{round(memory / (1024 * 1024), 3)} MB")

                # Finding the number of categorical features.
                categorical_features = len(self.categorical)
                data_dict["Categorical Features"] = categorical_features

                # Finding the number of numerical features.
                numerical_features = len(self.numerical)
                data_dict["Numerical Features"] = numerical_features

                # Finding the feature histogram and storing as a BytesIO object.
                pandas_df.hist(figsize=(20, 20))

                hist_img = io.BytesIO()
                plt.savefig(hist_img, format="png")
                plt.close()
                hist_img.seek(0)

                data_dict["Histogram"] = hist_img

                write_logs(
                    error=False,
                    message=f"Successfully Found Feature Histogram"
                )

                return data_dict

        except Exception as BasicDataError:
            write_logs(
                error_reason=BasicDataError,
                error=True,
                message=f"Could Not Find Basic Knowledge About The Dataset Due To {BasicDataError}"
            )

    def column_data(self, is_categorical: bool, column_index: int) -> dict:
        """
        The function finds all the data about a particular column in the dataset.
        :param is_categorical: Pass True if the column in categorical.
        :param column_index: The index of the column in the dataset.
        :return: A dictionary containing all the data pertaining to the column.
        """
        try:
            column_dict = dict()
            pandas_dataframe = self.df.compute()

            if not is_categorical:
                # Find column by index.
                column = pandas_dataframe.iloc[:, column_index]

                # Find missing values.
                missing = column.isnull().sum().sum()
                column_dict["Missing"] = missing

                # Find unique values.
                unique = column.unique()
                column_dict["Unique"] = len(unique)

                # Find percentiles.
                percentiles = [i for i in range(0, 101)]

                # Finding normal percentiles.
                percentile_data = np.percentile(column, percentiles)

                # Theoretical Normal Distribution.
                normal_percentiles = norm.ppf(np.array(percentiles) / 100.0)

                # Generating a graph object.
                plt.figure(figsize=(8, 6))
                plt.scatter(normal_percentiles, percentile_data, color='blue', alpha=0.6)
                plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')  # Diagonal line for reference
                plt.title('Percentile Plot (Q-Q Plot)')
                plt.xlabel('Theoretical Quantiles (Normal Distribution)')
                plt.ylabel('Sample Quantiles (Data)')
                plt.grid(True)

                # Saving the graph in BytesIO type object.
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                plt.close()
                img_data.seek(0)

                # Adding to the dictionary.
                column_dict["Graph"] = img_data

                # Adding column description to the dictionary.
                description_dict = dict()
                for key, value in dict(column.describe()).items():
                    description_dict[key] = value
                column_dict["Description"] = description_dict

                write_logs(
                    error=False,
                    message="Perfectly Extracted Column Wise Details."
                )

                return column_dict
            else:
                # Finding the column by index.
                column = pandas_dataframe.iloc[:, column_index]

                # Find values in the column.
                column_dict["Columns"] = len(column)

                # Find missing values.
                missing = column.isnull().sum().sum()
                column_dict["Missing"] = missing

                # Find unique values.
                unique = column.unique()
                column_dict["Unique"] = len(unique)

                # Counting the occurrences of each category.
                count_category = column.value_counts()

                # Generating a pie chart.
                plt.figure(figsize=(10, 6))
                plt.pie(count_category, labels=count_category.index, autopct='%1.1f%%', startangle=140)

                # Saving the graph in BytesIO type object.
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                plt.close()
                img_data.seek(0)

                # Adding to the dictionary.
                column_dict["Graph"] = img_data

                # Adding column description to the dictionary.
                description_dict = dict()
                for key, value in dict(column.describe()).items():
                    description_dict[key] = value
                column_dict["Description"] = description_dict

                write_logs(
                    error=False,
                    message="Perfectly Extracted Column Wise Details."
                )

                return column_dict

        except Exception as ColumnDataError:
            write_logs(
                error_reason=ColumnDataError,
                error=True,
                message=f"Could Not Find Column Related Details For Index {column_index} Due To {ColumnDataError}"
            )
