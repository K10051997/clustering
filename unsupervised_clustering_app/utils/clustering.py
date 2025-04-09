import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd

def run_kmeans(data, k):
    model = KMeans(n_clusters=k, random_state=10)
    labels = model.fit_predict(data)
    return labels, model

def visualize_clusters(data, labels):
    df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
    df['Cluster'] = labels
    
    fig, ax = plt.subplots()
    sns.scatterplot(x='Feature1', y='Feature2', hue='Cluster', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)
