import streamlit as st
import pandas as pd
import dask.dataframe as dd

st.title("Upload Your Dataset")

st.markdown(
    """
    You can upload any .csv or .xlsx file, but there are some constraints:
    - **Target:** Make sure your dataset has only one target variable.
    - **Header:** Make sure your file has its first column as column headers.
    - **Data:** Make sure that there is at least one categorical feature in the
    dataset.
    """
)


# Function to handle file upload.
def upload_file():
    uploaded_file = st.file_uploader("Choose Your Dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.success("File Uploaded Successfully")
        st.session_state["data"] = uploaded_file


# Function to handle target index input.
def input_target_index():
    target = st.number_input("Enter The Column Index Of The Target", min_value=0, step=1, key="target_index")
    st.session_state["index"] = target


# Call the upload function.
upload_file()

# Check if a file has been uploaded.
if "data" in st.session_state:
    # Adding file to the session state.
    uploaded_file = st.session_state["data"]

    # Call the target index input function
    input_target_index()

    # Check if target index is in the session state
    if "index" in st.session_state:
        target_index = st.session_state["index"]

        if uploaded_file.type == "text/csv":
            dataframe = pd.read_csv(uploaded_file)
            st.session_state["data"] = dd.from_pandas(dataframe)
            st.write("Here's a preview of your dataset:")
            st.dataframe(dataframe.head())
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            dataframe = pd.read_excel(uploaded_file)
            st.session_state["data"] = dd.from_pandas(dataframe)
            st.write("Here's a preview of your dataset:")
            st.dataframe(dataframe.head())
