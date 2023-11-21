#!/usr/bin/env python
# coding: utf-8

# ## Importing Modules

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df =  pd.read_csv(r"labeled_data.csv").drop(columns=['Unnamed: 0'])

# Streamlit code
st.title("Tweet Analysis")

# Display the question
st.header("Out of all the tweets, in which category is a tweet classified?")

# Visualization
category_count = df["class"].map({0: "Hate_Speech", 1: "Offensive_Language", 2: "Neither"}).value_counts()
fig, ax = plt.subplots()
category_count.plot(kind="bar", ax=ax)
plt.title("Tweet Category Vs Frequency")
plt.xlabel("Category")
plt.ylabel("Frequency")
st.pyplot(fig)

st.header("DataFrame head:")


# Display the DataFrame
st.dataframe(df.head())
