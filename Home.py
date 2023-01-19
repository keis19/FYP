#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

from collections import Counter
from datetime import datetime

# Visualization
# import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import average_precision_score, precision_recall_curve
# from sklearn.metrics import auc, plot_precision_recall_curve
# from sklearn.metrics import roc_auc_score

import streamlit as st


# In[7]:


st.set_page_config(
  page_title= "Something's Fishy",
  page_icon="🎣",
  layout="wide"
)

st.title("Something's Fishy")
st.header("We'll help you discover if your data contains any fraud cases. 🚨")
st.sidebar.success("Select a page.")

st.markdown(
    """
   Welcome to the home page of Something's Fishy. We help you to detect fraud cases from your dataset using our Machine Learning Algorithm.
   Upload your file to get started !
"""
)


# In[4]:


# uploaded_file = st.file_uploader("Choose a file")
# df_train=pd.read_csv(uploaded_file)
# # df_train= pd.read_csv("D:/Users/Acer/Desktop/Y3S1/Final Year Project/Datasets/fraudTrain.csv")
# df_train = df_train.drop(df_train.columns[0], axis=1)
# st.subheader("Data: ")
# st.table(df_train.head())

def process_file(uploaded_file):
    try:
        df_train=pd.read_csv(uploaded_file)
        st.session_state['df_train']=df_train
    except Exception as e:
        st.error("An error occurred while loading the file: " + str(e))
     
    
    df_train = df_train.drop(df_train.columns[0], axis=1)
    st.subheader("Data: ")
    st.table(df_train.head())

    df_train.rename(columns={"trans_date_trans_time":"transaction_time",
                             "cc_num":"credit_card_number",
                             "amt":"amount(usd)",
                             "trans_num":"transaction_id"},
                    inplace=True)

    df_train["transaction_time"] = pd.to_datetime(df_train["transaction_time"], infer_datetime_format=True)
    df_train["dob"] = pd.to_datetime(df_train["dob"], infer_datetime_format=True)

    df_train['time'] = df_train['unix_time'].apply(datetime.utcfromtimestamp)
    df_train.drop('unix_time', axis=1)
    df_train['hour_of_day'] = df_train.time.dt.hour

    df_train.credit_card_number = df_train.credit_card_number.astype('category')
    df_train.is_fraud = df_train.is_fraud.astype('category')
    df_train.hour_of_day = df_train.hour_of_day.astype('category')

    features = ['credit_card_number', 'merchant', 'category',
           'amount(usd)', 'zip',  'city_pop', 'job', 'transaction_id',
           'unix_time','hour_of_day', 'merch_lat', 'merch_long','lat' ,'long']

    X = df_train[features].set_index("transaction_id")
    y = df_train['is_fraud']

    enc = OrdinalEncoder(dtype=np.int64)
    enc.fit(X.loc[:,['category','merchant','job']])

    X.loc[:, ['category','merchant','job']] = enc.transform(X[['category','merchant','job']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    smt = SMOTE(sampling_strategy=0.1)
    X_train_smote, y_train_smote = smt.fit_resample(X_train.astype('float'), y_train)

    KNC= KNeighborsClassifier()
    KNC.fit(X_train_smote, y_train_smote)
    y_pred = KNC.predict(X_test)
    print(classification_report(y_test,y_pred))
    
    cm=confusion_matrix(y_test,y_pred)
    tn, fp, fn,tp = cm.ravel()
    print("Correctly predicted fraud cases: ", tp)
    st.write("There are a total of: ", tp ,"cases")
    
    
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.error("Please upload a file")
else:
    process_file(uploaded_file)
    
    # uploaded = False
# if not uploaded:
#     uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
#     uploaded = True
# if uploaded_file is None:
#     st.error("Please upload a file")
# else:
#     try:
#         df_train=pd.read_csv(uploaded_file)
#     except Exception as e:
#         st.error("An error occurred while loading the file: " + str(e))


# In[5]:


# df_train.rename(columns={"trans_date_trans_time":"transaction_time",
#                          "cc_num":"credit_card_number",
#                          "amt":"amount(usd)",
#                          "trans_num":"transaction_id"},
#                 inplace=True)


# # In[6]:


# df_train["transaction_time"] = pd.to_datetime(df_train["transaction_time"], infer_datetime_format=True)
# df_train["dob"] = pd.to_datetime(df_train["dob"], infer_datetime_format=True)


# # In[7]:


# # Apply function utcfromtimestamp and drop column unix_time
# df_train['time'] = df_train['unix_time'].apply(datetime.utcfromtimestamp)
# df_train.drop('unix_time', axis=1)

# # Add column hour of day
# df_train['hour_of_day'] = df_train.time.dt.hour


# # In[8]:


# # Change dtypes
# df_train.credit_card_number = df_train.credit_card_number.astype('category')
# df_train.is_fraud = df_train.is_fraud.astype('category')
# df_train.hour_of_day = df_train.hour_of_day.astype('category')


# # In[9]:


# np.round(df_train.describe(), 2)


# # In[10]:


# features = ['credit_card_number', 'merchant', 'category',
#        'amount(usd)', 'zip',  'city_pop', 'job', 'transaction_id',
#        'unix_time','hour_of_day', 'merch_lat', 'merch_long','lat' ,'long']

# X = df_train[features].set_index("transaction_id")
# y = df_train['is_fraud']

# print('X shape:{}\ny shape:{}'.format(X.shape,y.shape))


# enc = OrdinalEncoder(dtype=np.int64)
# enc.fit(X.loc[:,['category','merchant','job']])

# X.loc[:, ['category','merchant','job']] = enc.transform(X[['category','merchant','job']])


# # In[11]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
# print('X_train shape:{}\ny_train shape:{}'.format(X_train.shape,y_train.shape))
# print('X_test shape:{}\ny_test shape:{}'.format(X_test.shape,y_test.shape))


# # In[12]:


# #SMOTE
# smt = SMOTE(sampling_strategy=0.1)
# X_train_smote, y_train_smote = smt.fit_resample(X_train.astype('float'), y_train)
# print("Before SMOTE:", Counter(y_train))
# print("After SMOTE:", Counter(y_train_smote))


# # In[14]:


# KNC= KNeighborsClassifier()
# KNC.fit(X_train_smote, y_train_smote)
# y_pred = KNC.predict(X_test)
# print(classification_report(y_test,y_pred))

# # # Data to plot precision - recall curve
# # precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# # # Use AUC function to calculate the area under the curve of precision recall curve
# # auc_precision_recall = auc(recall, precision)
# # # plt.plot(recall, precision)
# # # fig1=plt.title('K Neighbors with SMOTE')
# # # fig1=plt.show()
# # # fig1

# # roc_auc = roc_auc_score(y_test, y_pred)
# # print("AUC-PR (SMOTE): %.3f " % auc_precision_recall)
# # print("ROC AUC (SMOTE): %.3f" % roc_auc)


# # In[16]:


# # fig2=plt.figure(figsize=(8,6))
# cm=confusion_matrix(y_test,y_pred)
# tn, fp, fn,tp = cm.ravel()
# print("Correctly predicted fraud cases: ", tp)
# st.subheader("There are a total of: ", tp ,"cases")

# # sns.heatmap(cm, cmap='viridis', annot=True, fmt='d', annot_kws=dict(fontsize=14))


# # In[ ]:
 





