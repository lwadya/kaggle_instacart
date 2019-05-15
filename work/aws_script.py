# Janky xgboost fix
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas.io.sql as pd_sql
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from code.lw_pickle import var_to_pickle
from code.lw_val_by_group import train_test_by_group


# Connect to SQL Database
connection_args = {
    #'host': '34.206.216.187',
    #'user': 'ubuntu',
    'dbname': 'instacart',
    #'port': 5432
}

connection = pg.connect(**connection_args)
csr = connection.cursor()
print('Connected to SQL Database')

# Read SQL Data
# Reads from Train Orders Table
cols = ['product_id',
        'add_to_cart_order',
        'user_id',
        'order_number',
        'order_dow',
        'order_hour_of_day',
        'days_since_prior_order']
cols = ', '.join(cols)
query = '''
    SELECT orderstrain.order_id, %s
    FROM orderstrain
    INNER JOIN orders ON orderstrain.order_id = orders.order_id
    ;
''' % cols
orders_train_df = pd_sql.read_sql(query, connection)
# Reads from Prior Orders Table
cols = ['product_id',
        'add_to_cart_order',
        'user_id',
        'order_number',
        'order_dow',
        'order_hour_of_day',
        'days_since_prior_order']
cols = ', '.join(cols)
query = '''
    SELECT ordersprior.order_id, %s
    FROM ordersprior
    INNER JOIN orders ON ordersprior.order_id = orders.order_id
    ;
''' % cols
orders_prior_df = pd_sql.read_sql(query, connection)
print('Read SQL Tables to DataFrames')

# Merge Cart-level DataFrame
df = (orders_prior_df.groupby(['product_id', 'user_id'], as_index=False)
                     .agg({'order_id':'nunique'})
                     .rename(columns={'order_id':'count_in_user_orders'}))
train_users = orders_train_df['user_id'].unique()
df = df[df['user_id'].isin(train_users)]
df.reset_index(drop=True, inplace=True)
train_users = None
print('Merged Cart-level DataFrame')

# Feature Engineering
# Target: Whether or Not Product is in Cart
train_carts_df = (orders_train_df.groupby('user_id')
                                 .agg({'product_id':(lambda x: set(x))})
                                 .rename(columns={'product_id':'cart_contents'}))

df = df.merge(train_carts_df, on='user_id')
df['in_cart'] = (df.apply(lambda row: row['product_id'] in row['cart_contents'], axis=1)
                   .astype(int))
df.drop('cart_contents', axis=1, inplace=True)
train_carts_df = None
# Feature: Product Order Rate by User
prior_per_user_df = (orders_prior_df.groupby(['user_id'])[['order_id']]
                                    .nunique()
                                    .rename(columns={'order_id':'total_user_orders'}))
df = df.merge(prior_per_user_df, on='user_id')
df['percent_in_user_orders'] = df['count_in_user_orders'] / df['total_user_orders']
df.drop(['count_in_user_orders'], axis=1, inplace=True)
prior_per_user_df = None
# Feature: Product Overall Order Rate
product_prior_df = (orders_prior_df.groupby(['product_id'], as_index=False)
                                   .agg({'order_id':'nunique'})
                                   .rename(columns={'order_id':'count_in_all_orders'})
                                   .sort_values(by=['count_in_all_orders'], ascending=False)
                                   .reset_index(drop=True))
num_orders = orders_prior_df['order_id'].nunique()
product_prior_df['percent_in_all_orders'] = (product_prior_df['count_in_all_orders'] /
                                             num_orders)
product_prior_df.drop('count_in_all_orders', axis=1, inplace=True)
df = df.merge(product_prior_df, on='product_id')
product_prior_df = num_orders = None
# Feature: In Last Cart
mask = (orders_prior_df.sort_values(by='order_number')
                       .groupby(['user_id'])['order_id']
                       .last())
last_contents_df = (orders_prior_df[orders_prior_df['order_id'].isin(mask)]
                    .groupby(['user_id'])['product_id'].unique())
last_contents_df = pd.DataFrame(last_contents_df)
last_contents_df.rename(columns={'product_id':'last_cart_contents'}, inplace=True)

df = df.merge(last_contents_df, how='left', on='user_id')
df['in_last_cart'] =\
    (df.apply(lambda row: row['product_id'] in row['last_cart_contents'], axis=1)
       .astype(int))
df.drop('last_cart_contents', axis=1, inplace=True)
last_contents_df = None
# Feature: Days/Orders Between Orders, Times Product Appears in Last 5 Orders
since_first_df = (orders_prior_df.groupby(['user_id', 'order_id'], as_index=False)
                                 .agg({'order_number':'first',
                                       'days_since_prior_order':'first'}))
since_first_df = (since_first_df.drop('days_since_prior_order', axis=1)
                                .merge(since_first_df.drop('order_id', axis=1),
                                                           how='left',
                                                           on='user_id'))

mask = since_first_df['order_number_x'] <= since_first_df['order_number_y']
since_newest_df = since_first_df[mask].drop(['user_id', 'order_number_y'], axis=1)
since_newest_df = (since_newest_df.groupby(['order_id', 'order_number_x'],
                                           as_index=False)['days_since_prior_order'].sum())
since_newest_df.drop('order_number_x', axis=1, inplace=True)
since_newest_df.rename(columns={'days_since_prior_order':'days_since_newest'}, inplace=True)

mask = since_first_df['order_number_x'] >= since_first_df['order_number_y']
since_first_df = since_first_df[mask].drop(['user_id', 'order_number_y'], axis=1)
since_first_df = (since_first_df.groupby(['order_id', 'order_number_x'],
                                         as_index=False)['days_since_prior_order'].sum())
since_first_df.rename(columns={'days_since_prior_order':'days_since_first_order',
                               'order_number_x':'order_number'}, inplace=True)

newest_cart_df =\
    (orders_train_df.groupby(['user_id'])[['order_number', 'days_since_prior_order']]
                    .first()
                    .rename(columns={'order_number':'newest_order_number'}))

orders_since_df = orders_prior_df[['user_id', 'order_id', 'product_id']]
orders_since_df = orders_since_df.merge(since_first_df, how='left', on='order_id')
orders_since_df = orders_since_df.merge(since_newest_df, how='left', on='order_id')
orders_since_df = orders_since_df.merge(newest_cart_df, how='left', on='user_id')
since_first_df = since_newest_df = newest_cart_df = None

orders_since_df['days_since_newest'] += orders_since_df['days_since_prior_order']
orders_since_df['orders_since_newest'] = (orders_since_df['newest_order_number'] -
                                          orders_since_df['order_number'])

mask = orders_since_df['order_number'] >= (orders_since_df['newest_order_number'] - 5)
last_five_df =\
    (orders_since_df[mask].groupby(['user_id', 'product_id'], as_index=False)['order_id']
                          .count()
                          .rename(columns={'order_id':'in_last_five'}))
orders_since_df = orders_since_df.merge(last_five_df,
                                        how='left',
                                        on=['user_id', 'product_id'])
orders_since_df['in_last_five'] = orders_since_df['in_last_five'].fillna(0).astype(int)
last_five_df = None

orders_since_df.sort_values(by=['user_id', 'product_id', 'order_number'], inplace=True)
orders_since_df.reset_index(drop=True, inplace=True)

orders_since_df['last_order_number'] =\
    orders_since_df.groupby(['user_id', 'product_id'])['order_number'].shift(1)
orders_since_df['last_days_since_first_order'] =\
    orders_since_df.groupby(['user_id', 'product_id'])['days_since_first_order'].shift(1)
orders_since_df['mean_orders_between'] =\
    orders_since_df['order_number'] - orders_since_df['last_order_number']
orders_since_df['mean_days_between'] =\
    orders_since_df['days_since_first_order'] - orders_since_df['last_days_since_first_order']

(orders_since_df['mean_orders_between'].fillna(orders_since_df['orders_since_newest'],
                                               inplace=True))
(orders_since_df['mean_days_between'].fillna(orders_since_df['days_since_newest'],
                                             inplace=True))

orders_since_df = (orders_since_df.groupby(['user_id', 'product_id'], as_index=False)
                                  .agg({'mean_orders_between':'mean',
                                        'mean_days_between':'mean',
                                        'orders_since_newest':'last',
                                        'days_since_newest':'last',
                                        'in_last_five':'last'}))
orders_since_df.rename({'order_number':'lastest_order_number'}, inplace=True)

df = df.merge(orders_since_df, how='left', on=['user_id', 'product_id'])
orders_since_df = None
# Feature: Likelihood a Product Gets Reordered
product_proba_df = (orders_prior_df.groupby('product_id', as_index=False)
                                   .agg({'user_id':'nunique', 'order_id':'count'}))
product_proba_df['product_reorder_proba'] = 1 - (product_proba_df['user_id'] /
                                                 product_proba_df['order_id'])
product_proba_df.drop(['user_id', 'order_id'], axis=1, inplace=True)
df = df.merge(product_proba_df, how='left', on='product_id')
product_proba_df = None
# Feature: Likelihood a User Reorders Any Product
user_proba_df = (orders_prior_df.groupby('user_id', as_index=False)
                                .agg({'product_id':'nunique', 'order_id':'count'}))
user_proba_df['user_reorder_proba'] = 1 - (user_proba_df['product_id'] /
                                           user_proba_df['order_id'])
user_proba_df.drop(['product_id', 'order_id'], axis=1, inplace=True)
df = df.merge(user_proba_df, how='left', on='user_id')
product_proba_df = None
# Feature: Average Hour of Week, Order Size, and Add Order Percentile for Prior Orders
hour_of_week_df = (orders_prior_df.groupby('order_id')
                                  .agg({'order_dow':'first',
                                        'order_hour_of_day':'first',
                                        'add_to_cart_order':'max'})
                                  .rename(columns={'add_to_cart_order':'mean_cart_size'}))
hour_of_week_df['mean_hour_of_week'] = (hour_of_week_df['order_dow'] * 24 +
                                        hour_of_week_df['order_hour_of_day'])
hour_of_week_df.drop(['order_dow', 'order_hour_of_day'], axis=1, inplace=True)

percentile_df = (orders_prior_df[['user_id', 'order_id', 'product_id', 'add_to_cart_order']]
                 .merge(hour_of_week_df, how='left', on='order_id'))
percentile_df['mean_cart_percentile'] = (1 - (percentile_df['add_to_cart_order'] - 1) /
                                         percentile_df['mean_cart_size'])
percentile_df = (percentile_df.groupby(['user_id', 'product_id'])
                              .agg({'mean_cart_size':'mean',
                                    'mean_cart_percentile':'mean',
                                    'mean_hour_of_week':'mean'}))

df = df.merge(percentile_df, how='left', on=['user_id', 'product_id'])
hour_of_week_df = percentile_df = None
# Feature: Hour of Week and Number of Items in Newest Order
hour_of_week_df = (orders_train_df.groupby('user_id')
                                  .agg({'order_dow':'first',
                                        'order_hour_of_day':'first',
                                        'add_to_cart_order':'max'})
                                  .rename(columns={'add_to_cart_order':'newest_cart_size'}))
hour_of_week_df['newest_hour_of_week'] = (hour_of_week_df['order_dow'] * 24 +
                                          hour_of_week_df['order_hour_of_day'])
hour_of_week_df.drop(['order_dow', 'order_hour_of_day'], axis=1, inplace=True)

df = df.merge(hour_of_week_df, how='left', on=['user_id'])
hour_of_week_df = None
# Feature: Absolute Difference in Cart Size, Hour of Week, Hour, and Day
df['cart_size_difference'] = np.abs(df['mean_cart_size'] - df['newest_cart_size'])
df['hour_of_week_difference'] = np.abs(df['mean_hour_of_week'] - df['newest_hour_of_week'])
print('Engineered Features')

# Split Into Train, Validate, and Test Sets
group_col = 'user_id'
x_cols = ['percent_in_user_orders',
          'percent_in_all_orders',
          'in_last_cart',
          'in_last_five',
          'total_user_orders',
          'mean_orders_between',
          'mean_days_between',
          'orders_since_newest',
          'days_since_newest',
          'product_reorder_proba',
          'user_reorder_proba',
          'mean_cart_size',
          'mean_cart_percentile',
          'mean_hour_of_week',
          'newest_cart_size',
          'newest_hour_of_week',
          'cart_size_difference',
          'hour_of_week_difference'
         ]
y_col = 'in_cart'

train_df, test_df = train_test_by_group(df, group_col, test_size=.1)
print('Created Train-Test Split')

# Scale Features
scl = MinMaxScaler()
X_train = scl.fit_transform(train_df[x_cols].values)
X_test = scl.transform(test_df[x_cols].values)
test_df.reset_index(drop=True, inplace=True)
for name, col in zip(x_cols, np.transpose(X_test)):
    test_df.loc[:, name] = col
print('Scaled Features')

# Logistic Regression
lr = LogisticRegression(C=10, solver='lbfgs', multi_class='auto', max_iter=2000)
lr.fit(X_train, train_df[y_col])
print('Trained Logistic Regression')

# XGBoost
gbm = xgb.XGBClassifier(n_estimators=10000,
                        max_depth=3,
                        objective="binary:logistic",
                        learning_rate=.5,
                        subsample=.08,
                        min_child_weight=.5,
                        colsample_bytree=.8)
gbm.fit(X_train, train_df[y_col],
        eval_set=[(X_train, train_df[y_col])],
        eval_metric='auc',
        early_stopping_rounds=20,
        verbose=False)
print('Trained XGBoost')

# Save Pickles
test_df.to_csv('pickle/test_df.csv')
var_to_pickle(gbm, 'pickle/model_gbm.pk')
var_to_pickle(lr, 'pickle/model_lr.pk')
print('Saved Pickle Files')
