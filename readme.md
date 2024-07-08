# DataWhisperer - Interact With Your Dataset
<hr />

DataWhisperer is an open-source website tool developed in
Python. It is used to automate any classic ML task like 
regression, classification, clustering, dimensionality
reduction, or ranking. It can automate basic data cleaning
and preprocessing techniques, provide all required data
visualization column-wise, find the best model for the
dataset, train the model, and display the training and
testing results.

The most interesting feature in the website is the ability
to generate fundamental patterns and insights about the
dataset.

## Features
- Automate the process of data preprocessing.
- Perform basic visualizing tasks, to check for various feature properties.
- Find and select the best model for the task and the dataset.
- Train, test and validation the accuracy for the model.
- Find basic insights like linear seperability, feature importance.
- Advanced insights like hierarchies, associate rules, frequent item-set mining.

## Testing Datasets
All datasets used for testing this project have been acquired from the below listed sources:
- **Loan Approval Dataset:** [Loan Appoval Dataset - Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Weather Prediction Dataset:** [Weather Classifcation Dataset - Kaggle](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)
- **HR Employee Analytics:** [HR Employee Analytics](https://www.kaggle.com/datasets/kmldas/hr-employee-data-descriptive-analytics)

## Sample Run
Here's a small sample run of the project:

https://github.com/K0MPLEXWorksTogether/DataWhisperer/assets/139705842/ac6672e5-b066-41d5-a389-8bc898c03b72


## Usage

Here's how you can use the app to your benefit:

- **Upload A Dataset:** Upload any .csv or .xlsx file, check the site for limitations.
- **Data Visualization:** You can view basic statistics of your dataset, and also column-wise statistics.
- **Data Preprocessing:** My code can finally generate completely data preprocessed training, testing and validation data splits back.
- **Model Generation:** *This part takes a little time.* The website finds the best model for your dataset, and display's the training, testing and validation classification report, along with the confusion matrix.
- **Fundamental Patterns And Insights:** You can check for basic insights like feature importance, linear separability, and some advance insights like associate rules, itemset mining, and hierarchies usnig a dendrogram.

## Limitations
While the tool itself is built on the idea of automating classic ML tasks, it has not reached that level of automation.

- **Classification Only:** The website for now can only deal with multiclass or binary classifcation. Work must be put in for regression, clustering, etc.
- **Exploratory Data Analysis:** As different problems and different datasets require different approaches, automation of EDA can help in deciding models and tasks.
- **Data Preprocessing:** A minimum of one text-based feature is required in the dataset for now.


## Future Scope
The future scope of this project is to bring in other classic ML problems like classification, regression and ranking.

Any contributions to its development are highly encouraged.

## Acknowledgements
This project is a complete brain child of 
[Abhiram Mangipudi](https://github.com/K0MPLEXWorksTogether/). If you like the work, please consider leaving a star, or coming forward for contributions.

## License
This project is released under the [MIT License](./LICENSE)