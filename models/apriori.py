import dask.dataframe as dd
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder


class Apriori:
    """
    The class performs the apriori algorithm, which is a very simple
    way to mine frequent patterns in the dataset. It provides two methods:
    - itemsets(): To get frequently occurring itemsets.
    - rules(): To mine associate rules from these frequent itemsets.
    """

    def __init__(self, dataframe: dd.DataFrame) -> None:
        """
        The constructor finds all the numerical columns in the dataset,
        and encodes them using the OneHotEncoder.
        """
        # Converting to pandas dataframe, as apriori does not support dask dataframes.
        pandas_df = dataframe.compute()

        # Finding numerical and categorical features.
        numerical_types = pandas_df.select_dtypes(include=["number"]).columns.tolist()

        categorical_types = [col for col in pandas_df.columns if col not in numerical_types]

        if categorical_types:
            self.categorical_data = pandas_df[categorical_types]

            # Creating OneHotEncoder.
            one_hot_encoder = OneHotEncoder(sparse_output=False)

            # Fitting and transforming on categorical data.
            encoded_data = one_hot_encoder.fit_transform(self.categorical_data)

            # Converting back to DataFrame and setting appropriate column names.
            self.categorical_data = pd.DataFrame(
                encoded_data,
                columns=one_hot_encoder.get_feature_names_out(categorical_types)
            )
        else:
            self.categorical_data = pd.DataFrame()

    def itemsets(self) -> pd.DataFrame:
        """
        The function finds the most frequently occurring patterns in the
        dataset, based on a tolerable support, and returns them.
        :return: A dataframe of itemsets.
        """
        if not self.categorical_data.empty:
            frequent = apriori(self.categorical_data, min_support=0.01, use_colnames=True)
            return frequent
        else:
            return pd.DataFrame(columns=['support', 'itemsets'])

    def rules(self) -> pd.DataFrame:
        """
        The function returns some associate rules on the dataset, based on
        some tolerable confidence, and returns them.
        :return: A Dataframe of rules.
        """
        itemsets = self.itemsets()

        rules = association_rules(itemsets, metric="confidence", min_threshold=0.01)
        return rules


def main():
    """
    Testing function. You can call main to test the functionality of the
    class.
    :return: None.
    """

    # Reading a csv file.
    data = dd.read_csv("../data/weather.csv")

    # Initializing the apriori object.
    apriori_obj = Apriori(data)
    print(apriori_obj.itemsets())
    print(apriori_obj.rules())


if __name__ == "__main__":
    main()
