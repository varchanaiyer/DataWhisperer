"""
The file is responsible for finding all the patterns and insights from the
dataframe. It utilizes all the classes from models for this task.
"""

import streamlit as st

from models.linear_seperability import LinearSeparability
from models.feature_importance import FeatureImportance
from models.apriori import Apriori
from models.dbscan import DBScan
from models.hierarchies import Hierarchies

st.title("Fundamental Patterns")
st.markdown(
    """
    Here, you can find some of the most basic patterns identified in 
    your dataset like:
    
    - Linear Seperability.
    - Important Features.
    
    Also, high-level insights like:
    - Frequently Occurring Itemsets With A Possible Support (`Apriori`).
    - Mined Associate Rules From The Dataset With Confidence (`Apriori`).
    - Clusters Within The Dataset (`DBSCAN`). 
    - Hierarchical Clusters Within The Dataset. (`scipy.cluster`)
    """
)

if "data" in st.session_state and "index" in st.session_state:
    st.subheader("Check For Linear Separability: ")
    linear_separable = LinearSeparability(st.session_state["data"], st.session_state["index"])
    is_linear_separable = linear_separable.prediction()

    for key, value in is_linear_separable.items():
        st.write(f"`{key}`: {value}")

    st.subheader("Feature Importance In The Dataset: ")
    feature_importance = FeatureImportance(st.session_state["data"], st.session_state["index"])
    with st.expander("Graph Of Feature Importance"):
        st.image(feature_importance.visualize_importance())

    apriori = Apriori(st.session_state["data"])
    st.subheader("Frequently Occurring Itemsets With Support:")
    with st.expander("A Dataframe Of Frequently Occurring Itemsets"):
        st.dataframe(apriori.itemsets())

    st.subheader("Associate Rules From The Dataset With Confidence:")
    with st.expander("A Dataframe Of Associate Rules From The Dataset"):
        st.dataframe(apriori.rules())

    st.subheader("Density Based Clusters In The Dataset:")
    dbscan = DBScan(st.session_state["data"], st.session_state["index"])
    dbscan.cluster()
    with st.expander("Scatter Plot Of Clusters"):
        st.image(dbscan.visualize())
        st.metric(label="Silhouette Score", value=dbscan.score())

    st.subheader("Hierarchical Clusters In The Dataset: ")
    hierarchy = Hierarchies(st.session_state["data"], st.session_state["index"])
    st.image(hierarchy.dendrogram())