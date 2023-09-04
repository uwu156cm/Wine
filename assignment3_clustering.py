#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Wine.csv')


# In[2]:


# Streamlit UI
st.title('Wine Clustering App')

# Sidebar controls
num_clusters = st.sidebar.slider('Select the number of clusters (K)', 2, 10, 3)


# In[3]:


def perform_clustering(data, num_clusters):
    # Select features for clustering
    X = data.drop('Customer_Segment', axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Assign cluster labels to the data
    cluster_labels = kmeans.labels_

    # Add cluster labels to the DataFrame
    data['Cluster'] = cluster_labels

    return data

# Perform clustering
clustered_data = perform_clustering(data, num_clusters)

# Display the clustered data
st.write('Clustered Data:')
st.write(clustered_data)


# In[ ]:




