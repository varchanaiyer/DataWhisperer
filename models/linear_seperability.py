from sklearn.svm import SVC
from src.preprocess import DataPreProcessor

import dask.dataframe as dd


def return_splits(dataframe, target_index) -> list:
    """
    Simplifies calling DataPreProcessor's preprocess() method.
    :param dataframe: The dataframe passed.
    :param target_index: The index of the target features in the dataframe.
    :return: None.
    """
    preprocess = DataPreProcessor(dataframe, target_index)
    result = preprocess.preprocess()
    return result


def approx(value1: float, value2: float, tolerance: float = 0.01) -> bool:
    """
    Responsible for making approximations in training and test accuracy
    closeness. Returns a boolean.
    :param value1: A floating point value.
    :param value2: Another floating point values.
    :param tolerance: The amount which can be considered as approximate.
    :return: A boolean returning whether both the values are approximates.
    """
    return abs(value2 - value1) <= tolerance


class LinearSeparability:
    """
    The class can determine of the dataset is linearly separable or
    not. It uses the Support Vector Classifier (SVC) for this.
    """

    def __init__(self, test_data: dd.DataFrame, index_target: int):
        # Splitting into train, test and validation
        train, test, validation = return_splits(test_data, index_target)

        # Finding the target column.
        target_column = train.columns[-1]

        # Separating features from targets.
        self.x_train = train.drop(columns=[target_column]).compute()
        self.y_train = train[target_column].compute()
        self.x_test = test.drop(columns=[target_column]).compute()
        self.y_test = test[target_column].compute()

        # Initializing SVC Classifiers, both linear nad non-linear.
        self.svm_linear = SVC(kernel="linear")
        self.svm_non_linear = SVC(kernel="rbf")

        # Fitting training data.
        self.svm_linear.fit(self.x_train, self.y_train)
        self.svm_non_linear.fit(self.x_train, self.y_train)

        # Finding training accuracy.
        self.train_score_linear = self.svm_linear.score(self.x_train, self.y_train)
        self.train_score_non_linear = self.svm_non_linear.score(self.x_train, self.y_train)

        # Finding test accuracy.
        self.test_score_linear = self.svm_linear.score(self.x_test, self.y_test)
        self.test_score_non_linear = self.svm_non_linear.score(self.x_test, self.y_test)

    def prediction(self) -> dict:
        """
        The function is responsible for displaying all the data
        responsible to check for linear and non-linear seperability.
        :return: A dictionary representing the data.
        """

        final_verdict = dict()
        final_verdict["Training Accuracy Of Linear SVM"] = self.train_score_linear
        final_verdict["Training Accuracy Of Non-Linear SVM"] = self.train_score_non_linear
        final_verdict["Test Accuracy Of Linear SVM"] = self.test_score_linear
        final_verdict["Test Accuracy Of Non-Linear SVM"] = self.test_score_non_linear

        # Adding the final verdict.
        if approx(self.train_score_linear, self.test_score_linear):
            final_verdict["Verdict"] = ("Dataset is linearly separable, as linear SVM has approximately equal train "
                                        "and test accuracy")
        else:
            final_verdict["Verdict"] = ("Dataset is not-linearly separable, as linear SVM does not has significant "
                                        "difference in train and test accuracy.")

        return final_verdict


def main():
    """
    Testing function. Run the program to call main() for testing.
    :return: None.
    """
    data = dd.read_csv("../data/weather.csv")

    linear = LinearSeparability(data, 10)
    print(linear.prediction())


if __name__ == "__main__":
    main()
