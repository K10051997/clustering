import streamlit as st
import pandas as pd
from utils.preprocessing import load_and_preprocess
from utils.clustering import run_kmeans, visualize_clusters

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Unsupervised Customer Segmentation")

# Load built-in data
@st.cache_data
def load_data():
    return pd.read_csv("unsupervised_clustering_app/data/mall_customers.csv")

df = load_data()
st.subheader("📊 Raw Data")
st.dataframe(df.head())

# Preprocessing
processed_data, scaled_data = load_and_preprocess(df)

# Clustering options
n_clusters = st.slider("Select number of clusters (K):", 2, 10, 5)

# Run KMeans and visualize
labels, model = run_kmeans(scaled_data, n_clusters)
st.subheader("📌 Cluster Visualization")
visualize_clusters(scaled_data, labels)
