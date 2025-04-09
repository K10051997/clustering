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
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    df = pd.DataFrame(data, columns=[f'Feature{i+1}' for i in range(data.shape[1])])
    df['Cluster'] = labels

    st.markdown(f"Visualizing clusters using **{df.columns[0]}** and **{df.columns[1]}**")

    # Only take first 2 features for visualization
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df.columns[0], y=df.columns[1],
        hue='Cluster', data=df, palette='Set2', ax=ax
    )
    st.pyplot(fig)
