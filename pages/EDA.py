import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from datetime import datetime

# Initialization

df_train = st.session_state['key']
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

df_ = df_train.groupby(by=[pd.Grouper(key="transaction_time", freq="1W"),
                           'is_fraud','category']).agg({"amount(usd)":'mean',"transaction_id":"count"}).reset_index()


fig = px.scatter(df_,
        x='transaction_time',
        y='amount(usd)',
        color='is_fraud',
        facet_col ='category',
        facet_col_wrap=3,
        facet_col_spacing=.04,
        color_discrete_map={0:'purple', 1:'orange'}
)

fig.update_layout(height=1400,
                  width=960,
                  legend=dict(title='Fraud?'),
                  plot_bgcolor='#303030'
                 )

fig.update_yaxes(title='Mean Amount (USD)', matches=None)
fig.update_layout(title='Mean Amount(USD) based on Category')
fig.update_traces(marker=dict(size=10))
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True, title=''))

st.subheader("Exploratory Data Analysis 1")
st.write(fig)
st.subheader("Explanation:")
st.write("From this we get an overview on mean amount spend on Fradulent and Non-Fradulent cases based on category. We can see that some of the categories like travel, health & fintess and grocery have low spending for fraulent cases. While categoris like shopping, and entertainment have high expenditure. ")

groups =['is_fraud','job']
df_ = df_train.groupby(by=groups).agg({"amount(usd)":'mean',"transaction_id":"count"}).fillna(0).reset_index()

# Top 10 jobs had most fraud transactions.
df_ = df_[df_.is_fraud==1].sort_values(by='transaction_id',
                                       ascending=False).drop_duplicates('job', keep='first').iloc[:10, :]

fig = px.bar(df_,
             y='job', x='transaction_id',
             color='amount(usd)',
             color_continuous_scale=px.colors.sequential.Blues,
             labels={'job':'Job title', 
                     'transaction_id': 'Number of fraud transactions'},
             category_orders = {"job": df_.job.values},
             width=960,
             height=600)

fig.update_layout(
    title=dict(
        text='The Amount (USD) among the top 10 jobs with most fraud transactions.'
    ),
    plot_bgcolor='#808080'
)

fig.update_coloraxes(
    colorbar=dict(
        title='Amount(usd) of transactions',
        orientation='h',
        x=1
    ),
    reversescale=True
)

st.subheader("Exploratory Data Analysis 2")
st.write(fig)
st.subheader("Explanation:")
st.write("From this we can get an overview of jobs with heavy fradulent transactions.")


# Specified list of 12 merchants with the highest number of transactions.
top10_merchants = df_train.merchant.value_counts()[:10]

df_ = df_train.groupby(by=[pd.Grouper(key="transaction_time", freq="1W"),'is_fraud',
                           'merchant']).agg({"amount(usd)":'mean',"transaction_id":"count"}).reset_index()

df_ = df_[df_.merchant.isin(top10_merchants.index)]

fig = px.line(df_,
        x='transaction_time',
        y='amount(usd)',
        color='is_fraud',
        facet_col ='merchant',
        facet_col_wrap=3,
        facet_col_spacing=.06,
        category_orders={'merchant': top10_merchants.index}, # order the subplots
        color_discrete_map={1:'green', 0:'red'},
        line_shape='hv'
)

fig.update_layout(height=1200,
                  width=960,
                  title='Top 10 merchants that have Highest Transactions per week',
                  legend=dict(title='Is fraud?'),
                  plot_bgcolor='#303030'
                 )

fig.update_yaxes(title='Mean Amount (USD)', matches=None)
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

st.write("Exploratory Data Analysis 3")
st.write(fig)
st.subheader("Explanation:")
st.write("From this we can get a see of the few of the merchants in which credit card fraud occurs rapidly. This is a good visual to help detect common places visited by fraudsters. Some of the merchants to keep an eye out within this dataset would the merchant Killback LLC, Boyer PLC and Kuhn LLC.")



groups = ['credit_card_number']
df_ = df_train.groupby(by=groups).agg({"amount(usd)":'mean',"transaction_id":"count"}).fillna(0).reset_index()
df_.sort_values('transaction_id', ascending=False, inplace=True)

df_ = df_train[df_train.is_fraud==1].groupby(by='hour_of_day').agg({'transaction_id':'count'}).reset_index()
fig = px.area(data_frame=df_,
       x='hour_of_day',
       y='transaction_id',
       labels={'transaction_id':'Number of transaction'},
       )

fig.update_layout(
    title=dict(
        text='Number of FRAUD transactions by hours of day'
    ),
    plot_bgcolor='#fafafa'
)

fig.update_xaxes(type='category')

st.subheader("Exploratory Data Analysis 3")
st.write(fig)
st.write("From this we can see the common hours of the day in which fradulents transactions take place. Based on the figure it can be seen that alot of fradulents transaction take place during night to midnight.")

df_ = df_train.groupby(by=[pd.Grouper(key="transaction_time", freq="1M"),
                           'is_fraud','category']).agg({"amount(usd)":'sum',"transaction_id":"count"}).reset_index()

fig = px.line(
    df_[df_.is_fraud==1],
    x='transaction_time',
    y='amount(usd)',
    color='category',
    # barmode='stack'
    # color_discrete_sequence=px.colors.qualitative.Dark24
)

fig.update_layout(height=600,
                  width=960,
                  legend=dict(title='Categories'),
                  plot_bgcolor='#fafafa'
                 )

st.subheader("Exploratory Data Analysis 4")
st.write(fig)
st.subheader("Explanation:")
st.write("From this visualization we can see the months with highest expenditure based on category. It can clearly be seen that food and dining has the highest amoun spent and it occurs during Late November. Gas and transport also has a high expenditure during the month of September")

