import streamlit as st

st.markdown(
    """
    # DataWhisperer - Interact with your dataset.
    
    DataWhisperer is a ML automation tool that can provide interactive
    building of classification models and generate feature related insights
    for your dataset.
    
    ## Key Features
    - **Upload Your Dataset:** Any .csv or .xlsx is supported, but with
    some constraints.
    - **Data Visualization:** Check and visualize everything, from the 
    most basic details to column-wise normalization checks.
    - **Data Preprocessing:** Your dataset is automatically checked for 
    missing values, duplicates and encodes categories and scales numerical data.
    - **Automatic Model Selection:** A search for the model with the best
    percent better than baseline score is done, and finally returns the best
    model.
    - **Model Interaction:** You can pass some data from your dataset, set
    aside for testing to interact with your model, and get the classification
    report on training, test and the confusion matrix.
    - **Dataset Insights:** Basic insights like feature importance, or linear
    seperability are performed, but also some advanced methods like associate
    rule mining, or hierarchical clusters, provide insights from your dataset.
    """
)

st.warning(
    """
    DataWhisperer is in development stage. It can handle only binary
    or multi-class classification problems, and their related datasets.
    """
)
