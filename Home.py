#!/usr/bin/env python
# coding: utf-8

# In[34]:


import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

import os
import sqlite3
import math
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Model
from scipy.stats import skew
# import yellowbrick
import sklearn
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import streamlit as st


# In[4]:

st.set_page_config(
  page title: "Something's Fishy"
  page_icon="🎣"
  layout="wide"
)

st.title("Something's Fishy")
st.header("We'll help you discover if your data contains any fraud cases")
st.sidebar.success("Select a page ⬆⬆.")

st.markdown(
    """
   Welcome to the home page of Somethig's Fishy. We help you to detect fraud cases from your dataset using our Machine Learning Algorithm.
   Upload your file to get started !
"""
)

uploaded_file = st.file_uploader("Choose a file")
df_train=pd.read_csv(uploaded_file)

# Reading csv files and drop the first column
# df_train= pd.read_csv("C:/Users/Theeveeyan/Downloads/Keis/fraudTrain.csv")
df_train = df_train.drop(df_train.columns[0], axis=1)

df_test= pd.read_csv("C:/Users/Theeveeyan/Downloads/Keis/fraudTest.csv")
df_test = df_test.drop(df_test.columns[0], axis=1)

df_train.head()


# In[5]:


df_train.rename(columns={"trans_date_trans_time":"transaction_time",
                         "cc_num":"credit_card_number",
                         "amt":"amount(usd)",
                         "trans_num":"transaction_id"},
                inplace=True)


# In[6]:


df_train["transaction_time"] = pd.to_datetime(df_train["transaction_time"], infer_datetime_format=True)
df_train["dob"] = pd.to_datetime(df_train["dob"], infer_datetime_format=True)


# In[7]:


from datetime import datetime

# Apply function utcfromtimestamp and drop column unix_time
df_train['time'] = df_train['unix_time'].apply(datetime.utcfromtimestamp)
df_train.drop('unix_time', axis=1)

# Add column hour of day
df_train['hour_of_day'] = df_train.time.dt.hour


# In[8]:


# Change dtypes
df_train.credit_card_number = df_train.credit_card_number.astype('category')
df_train.is_fraud = df_train.is_fraud.astype('category')
df_train.hour_of_day = df_train.hour_of_day.astype('category')


# In[9]:


np.round(df_train.describe(), 2)


# In[10]:


df_train.columns



# In[12]:


print(df_train['transaction_time'])


# In[13]:




# In[15]:


features = ['transaction_id', 'hour_of_day', 'category', 'amount(usd)', 'merchant', 'job', 'credit_card_number', 'city_pop']

# removed features related to location: lat, long , zip, street, city, state

#
X = df_train[features].set_index("transaction_id")
y = df_train['is_fraud']

print('X shape:{}\ny shape:{}'.format(X.shape,y.shape))
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype=np.int64)
enc.fit(X.loc[:,['category','merchant','job']])

X.loc[:, ['category','merchant','job']] = enc.transform(X[['category','merchant','job']])


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
print('X_train shape:{}\ny_train shape:{}'.format(X_train.shape,y_train.shape))
print('X_test shape:{}\ny_test shape:{}'.format(X_test.shape,y_test.shape))


# In[ ]:


# # split first
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# # then select features using the training set only
# selector = SelectKBest(k=25)
# X_train_selected = selector.fit_transform(X_train,y_train)

# # fit again a simple logistic regression
# lr.fit(X_train_selected,y_train)
# # select the same features on the test set, predict, and get the test accuracy:
# X_test_selected = selector.transform(X_test)
# y_pred = lr.predict(X_test_selected)
# accuracy_score(y_test, y_pred)
# # 0.52800000000000002


# In[17]:


from sklearn.feature_selection import chi2
f_p_values=chi2(X_train,y_train)
f_p_values
import pandas as pd
p_values=pd.Series(f_p_values[1])
p_values.index=X_train.columns
p_values.sort_index(ascending=False)


# In[20]:


import imblearn
from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train_smote, y_train_smote = smt.fit_resample(X_train.astype('float'), y_train)
print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_smote))


# In[21]:


from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print(classification_report(y_test,y_pred))
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

# Data to plot precision - recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision)
fig1=plt.show()
fig1

fig2=plt.figure(figsize=(8,6))
cfs_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(cfs_matrix, cmap='viridis', annot=True, fmt='d', annot_kws=dict(fontsize=14))


# In[22]:


from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier()
dtree.fit(X_train_smote, y_train_smote)
y_pred = dtree.predict(X_test)
print(classification_report(y_test,y_pred))
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

# Data to plot precision - recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision)
fig1=plt.show()
fig1

fig2=plt.figure(figsize=(8,6))
cfs_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(cfs_matrix, cmap='viridis', annot=True, fmt='d', annot_kws=dict(fontsize=14))


# In[23]:


print('Random Forest Algorithm')
from sklearn.ensemble import RandomForestClassifier

rf_random = RandomForestClassifier()
rf_random.fit(X_train, y_train)
y_pred = rf_random.predict(X_test)

# Print report
print(classification_report(y_test, y_pred))
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

# Data to plot precision - recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision)
fig1=plt.show()
fig1

fig2=plt.figure(figsize=(8,6))
cfs_matrix=confusion_matrix(y_test,y_pred)
st.write(sns.heatmap(cfs_matrix, cmap='viridis', annot=True, fmt='d', annot_kws=dict(fontsize=14)))


# In[24]:


print('SMOTE')
rf_random = RandomForestClassifier()
rf_random.fit(X_train_smote, y_train_smote)
y_pred = rf_random.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

# Data to plot precision - recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision)
fig1=plt.show()
fig1

fig2=plt.figure(figsize=(8,6))
cfs_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(cfs_matrix, cmap='viridis', annot=True, fmt='d', annot_kws=dict(fontsize=14))


# In[25]:


print("XG Boost Classifier")
from sklearn.ensemble import GradientBoostingClassifier
XG_random = GradientBoostingClassifier()

XG_random.fit(X_train, y_train)
y_pred = XG_random.predict(X_test)

# Print reprort
print(classification_report(y_test, y_pred))


# In[33]:


print("SMOTE- XG Boost Classifier")
from sklearn.ensemble import GradientBoostingClassifier
XG_random = GradientBoostingClassifier()

XG_random.fit(X_train_smote, y_train_smote)
y_pred = XG_random.predict(X_test)

# Print reprort
print(classification_report(y_test, y_pred))


# In[32]:


print("Logistic Regression")
from sklearn.linear_model import LogisticRegression

LR= LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

print(classification_report(y_test,y_pred))


# In[27]:


print("SMOTE- Logistic Regression")
from sklearn.linear_model import LogisticRegression

LR= LogisticRegression()
LR.fit(X_train_smote, y_train_smote)
y_pred = LR.predict(X_test)

print(classification_report(y_test,y_pred))


# In[28]:


print('KNC')
from sklearn.neighbors import KNeighborsClassifier
KNC= KNeighborsClassifier()
KNC.fit(X_train, y_train)
y_pred = KNC.predict(X_test)

print(classification_report(y_test,y_pred))


# In[29]:


print('SMOTE-KNC')
KNC= KNeighborsClassifier()
KNC.fit(X_train_smote, y_train_smote)
y_pred = KNC.predict(X_test)

print(classification_report(y_test,y_pred))


# In[ ]:


print("Hello")

