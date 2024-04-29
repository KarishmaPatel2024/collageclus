import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Load the pre-trained KMeans model
kmeans_model = pickle.load(open("k_model.pkl","rb"))

# Load the dataset
df = pd.read_csv("college.csv")

# Convert 'Private' column to binary (1 for 'Yes', 0 for 'No')
def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

# Streamlit app
def main():
    # Set background image using CSS in markdown
    st.markdown(
    """
    <style>
    .main{
        background-image: url('https://img.huffingtonpost.com/asset/574c64be160000ab02f94c83.jpeg?ops=scalefit_720_noupscale&format=webp');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.title("College Clustering Analysis")
    st.sidebar.title("Settings")

    # Display dataset
    if st.sidebar.checkbox("Show Dataset"):
        st.subheader("College Dataset")
        st.write(df)

    # Display descriptive statistics
    if st.sidebar.checkbox("Show Descriptive Statistics"):
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

    # Perform clustering
    if st.sidebar.checkbox("Perform Clustering"):
        st.subheader("Clustering Results")
        st.write("Cluster Centers:")
        st.write(kmeans_model.cluster_centers_)

        # Predict clusters
        df['Predicted_Cluster'] = kmeans_model.labels_

        st.write("Confusion Matrix:")
        cm = confusion_matrix(df['Cluster'], df['Predicted_Cluster'])
        st.write(cm)

        st.write("Classification Report:")
        report = classification_report(df['Cluster'], df['Predicted_Cluster'])
        st.write(report)

    # Plotting
    st.sidebar.subheader("Visualizations")
    plot_options = ["Scatter Plot - Room.Board vs. Grad.Rate",
                    "Scatter Plot - Outstate vs. F.Undergrad",
                    "Histogram - Outstate",
                    "Histogram - Grad.Rate"]
    plot_choice = st.sidebar.selectbox("Choose a plot", plot_options)

    if plot_choice == "Scatter Plot - Room.Board vs. Grad.Rate":
        st.subheader("Room.Board vs. Grad.Rate")
        scatter_plot = sns.scatterplot(x='Room.Board', y='Grad.Rate', hue='Private', data=df, palette='coolwarm')
        st.pyplot(scatter_plot.figure)

    elif plot_choice == "Scatter Plot - Outstate vs. F.Undergrad":
        st.subheader("Outstate vs. F.Undergrad")
        scatter_plot = sns.scatterplot(x='Outstate', y='F.Undergrad', hue='Private', data=df, palette='coolwarm')
        st.pyplot(scatter_plot.figure)

    elif plot_choice == "Histogram - Outstate":
        st.subheader("Histogram - Outstate")
        g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
        hist_plot = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
        st.pyplot(hist_plot.fig)

    elif plot_choice == "Histogram - Grad.Rate":
        st.subheader("Histogram - Grad.Rate")
        g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
        hist_plot = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
        st.pyplot(hist_plot.fig)

if __name__ == "__main__":
    main()
