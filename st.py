#!/usr/bin/env python
# coding: utf-8

# ## Importing Modules

# In[2]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
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
