import pandas as pd
import numpy as np
import os
import psycopg2 as pg
import pandas.io.sql as pd_sql
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from lw_pickle import var_to_pickle
from val_by_group import train_test_by_group


class build_models:
    '''
    Acquires instacart data from SQL database, engineers features, and uses data
    to train logistic regression and gradient boosting tree models. This script
    is designed to be run locally and remotely on a cloud instance, either
    through a Jupyter Notebook or the command line.
    '''

    # Sets variables for id, y, and X columns
    id_col = 'user_id'
    y_col = 'in_cart'
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
              'hour_of_week_difference']

    def __init__(self, user_limit=-1):
        '''
        Initializes variables and runs all steps from querying database through
        pickling trained models

        Args:
            user_limit (int): limits the query to the first user_limit users,
                              any value zero or less returns data for all users

        Returns:
            None
        '''
        self.user_limit = user_limit
        self.newest_df = None
        self.prior_df = None
        self.df = None
        self.train_df = None
        self.test_df = None
        self.lrm = None
        self.gbm = None
        print(self.run())

    def run(self):
        '''
        Runs all steps from querying database through pickling trained models

        Args:
            None

        Returns:
            str: success or error message
        '''
        if self.query_db():
            return 'Failed to create dataframes from SQL database'
        print('Queried SQL database and created dataframes')

        # Still need to fully implement error checking for these functions
        self.merge_cart_df()
        self.feature_target()
        self.feature_rate_by_user()
        self.feature_rate_overall()
        self.feature_in_last_cart()
        self.feature_time_since_order()
        self.feature_product_proba()
        self.feature_user_proba()
        self.feature_mean_order()
        self.feature_newest_order()
        print('Engineered all features')

        self.train_test_split()
        print('Split data into train and test sets')

        self.model_log_reg()
        print('Trained logistic regression model')

        self.model_grad_boost()
        print('Trained gradient boosting decision tree model')

        self.save_to_files()
        print('Saved test set and models to files')

        return 'Success'

    def query_db(self):
        '''
        Queries table data from SQL database set up on an AWS ec2 instance.
        Saves output to dataframes. Can be run on a local machine with access to
        the ec2 or on the ec2 itself.

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        # ec2 SQL database settings
        connection_args = {
            'host': '34.206.216.187',
            'user': 'ubuntu',
            'dbname': 'instacart',
            'port': 5432
        }
        local_args = {
            'dbname': connection_args['dbname']
        }

        # Columns to be collected from SQL tables
        query_cols = ['product_id',
                      'add_to_cart_order',
                      'user_id',
                      'order_number',
                      'order_dow',
                      'order_hour_of_day',
                      'days_since_prior_order']

        # First attempts to connect to local SQL database, then defaults to ec2
        try:
            connection = pg.connect(**local_args)
        except:
            try:
                connection = pg.connect(**connection_args)
            except:
                return True
        #csr = connection.cursor()

        # Creates string user limit string for end of query
        if self.user_limit > 0:
            query_end = 'WHERE user_id < %i;' % int(self.user_limit)
        else:
            query_end = ';'

        # Reads from Train Orders Table
        cols = ', '.join(query_cols)
        query = '''
            SELECT orderstrain.order_id, %s
            FROM orderstrain
            INNER JOIN orders ON orderstrain.order_id = orders.order_id
            %s
        ''' % (cols, query_end)
        self.newest_df = pd_sql.read_sql(query, connection)

        # Reads from Prior Orders Table
        query = '''
            SELECT ordersprior.order_id, %s
            FROM ordersprior
            INNER JOIN orders ON ordersprior.order_id = orders.order_id
            %s
        ''' % (cols, query_end)
        self.prior_df = pd_sql.read_sql(query, connection)

        return None

    def merge_cart_df(self):
        '''
        Merges table dataframes in preparation for feature engineering

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        df = (self.prior_df.groupby(['product_id', 'user_id'], as_index=False)
                           .agg({'order_id':'nunique'})
                           .rename(columns={'order_id':'count_in_user_orders'})
        )
        train_users = self.newest_df['user_id'].unique()
        df = df[df['user_id'].isin(train_users)]
        df.reset_index(drop=True, inplace=True)
        self.df = df
        return None

    def feature_target(self):
        '''
        Adds a dataframe column for the target: whether or not a product is in
        the cart

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df = (self.newest_df.groupby('user_id')
                                 .agg({'product_id':(lambda x: set(x))})
                                 .rename(columns={'product_id':'cart_contents'})
        )

        df = self.df.merge(feat_df, on='user_id')
        df['in_cart'] =\
        (df.apply(lambda row: row['product_id'] in row['cart_contents'], axis=1)
           .astype(int))
        df.drop('cart_contents', axis=1, inplace=True)
        self.df = df
        return None

    def feature_rate_by_user(self):
        '''
        Adds a dataframe column for the feature product order rate by user

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df = (self.prior_df.groupby(['user_id'])[['order_id']]
                                .nunique()
                                .rename(columns={'order_id':'total_user_orders'})
        )
        df = self.df.merge(feat_df, on='user_id')
        df['percent_in_user_orders'] = (df['count_in_user_orders'] /
                                        df['total_user_orders'])
        df.drop(['count_in_user_orders'], axis=1, inplace=True)
        self.df = df
        return None

    def feature_rate_overall(self):
        '''
        Adds a dataframe column for the feature overall product order rate

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df =\
        (self.prior_df.groupby(['product_id'], as_index=False)
                      .agg({'order_id':'nunique'})
                      .rename(columns={'order_id':'count_in_all_orders'})
                      .sort_values(by=['count_in_all_orders'], ascending=False)
                      .reset_index(drop=True)
        )
        num_orders = self.prior_df['order_id'].nunique()
        feat_df['percent_in_all_orders'] = (feat_df['count_in_all_orders'] /
                                            num_orders
        )
        feat_df.drop('count_in_all_orders', axis=1, inplace=True)
        df = self.df.merge(feat_df, on='product_id')
        self.df = df
        return None

    def feature_in_last_cart(self):
        '''
        Adds a dataframe column for the feature product was in last cart

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        mask = (self.prior_df.sort_values(by='order_number')
                             .groupby(['user_id'])['order_id']
                             .last()
        )
        feat_df = (self.prior_df[self.prior_df['order_id'].isin(mask)]
                   .groupby(['user_id'])['product_id'].unique()
        )
        feat_df = pd.DataFrame(feat_df)
        feat_df.rename(columns={'product_id':'last_cart_contents'}, inplace=True)

        df = self.df.merge(feat_df, how='left', on='user_id')
        df['in_last_cart'] =\
        (df.apply(lambda row: row['product_id'] in row['last_cart_contents'],
                  axis=1).astype(int)
        )
        df.drop('last_cart_contents', axis=1, inplace=True)
        self.df = df
        return None

    def feature_time_since_order(self):
        '''
        Adds a dataframe column for the following time-related features:
        * mean orders between product orders
        * mean days between product orders
        * number of orders since most recent product order
        * number of days since most recent product order
        * number of orders in the last five that include a product

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        # Creates a dataframe that has rows containing every user order with
        # every user days_since_prior_order value
        since_first_df =\
        (self.prior_df.groupby(['user_id', 'order_id'], as_index=False)
                      .agg({'order_number':'first',
                            'days_since_prior_order':'first'})
        )
        since_first_df =\
        (since_first_df.drop('days_since_prior_order', axis=1)
                       .merge(since_first_df.drop('order_id', axis=1),
                              how='left',
                              on='user_id')
        )

        # Splits off a new dataframe that has a column containing days since
        # the most recent order for every order
        mask = (since_first_df['order_number_x'] <=
                since_first_df['order_number_y']
        )
        since_newest_df =\
        since_first_df[mask].drop(['user_id', 'order_number_y'], axis=1)
        since_newest_df =\
        (since_newest_df.groupby(['order_id', 'order_number_x'], as_index=False)
        ['days_since_prior_order'].sum()
        )
        since_newest_df.drop('order_number_x', axis=1, inplace=True)
        since_newest_df.rename(columns={'days_since_prior_order':
                                        'days_since_newest'}, inplace=True)

        # Finishes since_first_df so that it has a column containing days since
        # the first order for every order
        mask = (since_first_df['order_number_x'] >=
                since_first_df['order_number_y']
        )
        since_first_df =\
        since_first_df[mask].drop(['user_id', 'order_number_y'], axis=1)
        since_first_df =\
        (since_first_df.groupby(['order_id', 'order_number_x'], as_index=False)
        ['days_since_prior_order'].sum()
        )
        since_first_df.rename(columns={'days_since_prior_order':
                                       'days_since_first_order',
                                       'order_number_x':'order_number'},
                                       inplace=True)

        # Creates a dataframe containing the most recent order number for every
        # user
        newest_cart_df =\
        (self.newest_df.groupby(['user_id'])[['order_number',
                                              'days_since_prior_order']]
                       .first()
                       .rename(columns={'order_number':'newest_order_number'})
        )

        # Creates a new dataframe that combines the original prior_df with the
        # newly-created dataframes containing days since first order, days since
        # most recent order, and most recent order number per user
        orders_since_df = self.prior_df[['user_id', 'order_id', 'product_id']]
        orders_since_df =\
        orders_since_df.merge(since_first_df, how='left', on='order_id')
        orders_since_df =\
        orders_since_df.merge(since_newest_df, how='left', on='order_id')
        orders_since_df =\
        orders_since_df.merge(newest_cart_df, how='left', on='user_id')

        orders_since_df['days_since_newest'] +=\
        orders_since_df['days_since_prior_order']

        orders_since_df['orders_since_newest'] =\
        (orders_since_df['newest_order_number'] -
        orders_since_df['order_number']
        )

        # Creates and merges a dataframe containing whether or not a product
        # appears in the last five user orders
        mask = (orders_since_df['order_number'] >=
                (orders_since_df['newest_order_number'] - 5)
        )
        last_five_df =\
        (orders_since_df[mask].groupby(['user_id', 'product_id'],
                                       as_index=False)['order_id']
                              .count()
                              .rename(columns={'order_id':'in_last_five'})
        )
        orders_since_df = orders_since_df.merge(last_five_df,
                                                how='left',
                                                on=['user_id', 'product_id'])
        orders_since_df['in_last_five'] =\
        orders_since_df['in_last_five'].fillna(0).astype(int)

        orders_since_df.sort_values(by=['user_id',
                                        'product_id',
                                        'order_number'], inplace=True)
        orders_since_df.reset_index(drop=True, inplace=True)

        # Adds columns to orders_since_df for the mean days between when an item
        # is reordered and the mean orders between when an item is reordered
        orders_since_df['last_order_number'] =\
        (orders_since_df.groupby(['user_id', 'product_id'])['order_number']
                        .shift(1)
        )
        orders_since_df['last_days_since_first_order'] =\
        (orders_since_df.groupby(['user_id', 'product_id'])
        ['days_since_first_order'].shift(1)
        )
        orders_since_df['mean_orders_between'] =\
        orders_since_df['order_number'] - orders_since_df['last_order_number']
        orders_since_df['mean_days_between'] =\
        (orders_since_df['days_since_first_order'] -
        orders_since_df['last_days_since_first_order']
        )

        (orders_since_df['mean_orders_between']
        .fillna(orders_since_df['orders_since_newest'], inplace=True)
        )
        (orders_since_df['mean_days_between']
        .fillna(orders_since_df['days_since_newest'], inplace=True)
        )

        # Merges new features back into the main dataframe
        orders_since_df =\
        (orders_since_df.groupby(['user_id', 'product_id'], as_index=False)
                        .agg({'mean_orders_between':'mean',
                              'mean_days_between':'mean',
                              'orders_since_newest':'last',
                              'days_since_newest':'last',
                              'in_last_five':'last'})
        )
        orders_since_df.rename({'order_number':'latest_order_number'},
                               inplace=True)

        df = self.df.merge(orders_since_df,
                           how='left',
                           on=['user_id', 'product_id'])
        self.df = df
        return None

    def feature_product_proba(self):
        '''
        Adds a dataframe column for the feature probability a product gets
        reordered

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df = (self.prior_df.groupby('product_id', as_index=False)
                                .agg({'user_id':'nunique', 'order_id':'count'}))
        feat_df['product_reorder_proba'] = 1 - (feat_df['user_id'] /
                                                feat_df['order_id'])
        feat_df.drop(['user_id', 'order_id'], axis=1, inplace=True)
        df = self.df.merge(feat_df, how='left', on='product_id')
        self.df = df
        return None

    def feature_user_proba(self):
        '''
        Adds a dataframe column for the feature probability a user reorders any
        product

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df =\
        (self.prior_df.groupby('user_id', as_index=False)
                      .agg({'product_id':'nunique', 'order_id':'count'})
        )
        feat_df['user_reorder_proba'] = 1 - (feat_df['product_id'] /
                                             feat_df['order_id'])
        feat_df.drop(['product_id', 'order_id'], axis=1, inplace=True)
        df = self.df.merge(feat_df, how='left', on='user_id')
        self.df = df
        return None

    def feature_mean_order(self):
        '''
        Adds a dataframe column for the following mean order features:
        * mean cart size for carts containing product
        * mean add order percentile for carts containing product
        * mean hour of week of orders containing product

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        # Calculates hour of week for each order
        hour_df = (self.prior_df.groupby('order_id')
                                .agg({'order_dow':'first',
                                      'order_hour_of_day':'first',
                                      'add_to_cart_order':'max'})
                                .rename(columns={'add_to_cart_order':
                                                 'mean_cart_size'})
        )
        hour_df['mean_hour_of_week'] = (hour_df['order_dow'] * 24 +
                                        hour_df['order_hour_of_day'])
        hour_df.drop(['order_dow', 'order_hour_of_day'], axis=1, inplace=True)

        # Calculates mean cart size, add order percentile, and order hour
        feat_df = (self.prior_df[['user_id',
                                  'order_id',
                                  'product_id',
                                  'add_to_cart_order']]
                       .merge(hour_df, how='left', on='order_id'))
        feat_df['mean_cart_percentile'] =\
        1 - (feat_df['add_to_cart_order'] - 1) / feat_df['mean_cart_size']
        feat_df = (feat_df.groupby(['user_id', 'product_id'])
                          .agg({'mean_cart_size':'mean',
                                'mean_cart_percentile':'mean',
                                'mean_hour_of_week':'mean'}))

        df = self.df.merge(feat_df, how='left', on=['user_id', 'product_id'])
        self.df = df
        return None

    def feature_newest_order(self):
        '''
        Adds a dataframe column for the following features:
        * hour of week of most recent order
        * items in most recent order
        * absolute difference in most recent and mean order cart size
        * absolute difference in most recent and mean order hour of week

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        feat_df = (self.newest_df.groupby('user_id')
                                 .agg({'order_dow':'first',
                                       'order_hour_of_day':'first',
                                       'add_to_cart_order':'max'})
                                 .rename(columns={'add_to_cart_order':
                                                  'newest_cart_size'})
        )
        feat_df['newest_hour_of_week'] = (feat_df['order_dow'] * 24 +
                                          feat_df['order_hour_of_day'])
        feat_df.drop(['order_dow', 'order_hour_of_day'], axis=1, inplace=True)

        df = self.df.merge(feat_df, how='left', on=['user_id'])
        df['cart_size_difference'] = np.abs(df['mean_cart_size'] -
                                            df['newest_cart_size'])
        df['hour_of_week_difference'] = np.abs(df['mean_hour_of_week'] -
                                               df['newest_hour_of_week'])
        self.df = df
        return None

    def train_test_split(self):
        '''
        Splits dataframe into train and test sets, removes unused predictor
        columns, and applies MinMax scaler to predictor values

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        train_df, test_df =\
        train_test_by_group(self.df, self.id_col, test_size=.1, seed=29)

        # Scales features
        scl = MinMaxScaler()
        X_train = scl.fit_transform(train_df[self.x_cols].values)
        X_test = scl.transform(test_df[self.x_cols].values)

        # Recombines feature-scaled training set dataframe
        train_df.reset_index(drop=True, inplace=True)
        train_df = train_df[[self.id_col, self.y_col]]
        temp_df = pd.DataFrame(X_train, columns=self.x_cols)
        self.train_df = pd.concat([train_df, temp_df], axis=1)

        # Recombines feature-scaled test set dataframe
        test_df.reset_index(drop=True, inplace=True)
        test_df = test_df[[self.id_col, self.y_col]]
        temp_df = pd.DataFrame(X_test, columns=self.x_cols)
        self.test_df = pd.concat([test_df, temp_df], axis=1)
        return None

    def model_log_reg(self):
        '''
        Trains a logistic regression model

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        self.lrm = LogisticRegression(C=10,
                                      solver='lbfgs',
                                      multi_class='auto',
                                      max_iter=2000)
        self.lrm.fit(self.train_df[self.x_cols], self.train_df[self.y_col])
        return None

    def model_grad_boost(self):
        '''
        Trains a gradient boosting decision tree model

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        self.gbm = xgb.XGBClassifier(n_estimators=10000,
                                     max_depth=3,
                                     objective="binary:logistic",
                                     learning_rate=.5,
                                     subsample=.08,
                                     min_child_weight=.5,
                                     colsample_bytree=.8)
        self.gbm.fit(self.train_df[self.x_cols], self.train_df[self.y_col],
                     eval_set=[(self.train_df[self.x_cols],
                                self.train_df[self.y_col])],
                     eval_metric='auc',
                     early_stopping_rounds=20,
                     verbose=False)
        return None

    def save_to_files(self):
        '''
        Saves test set to csv and models to pickle files

        Args:
            None

        Returns:
            bool: None if successful, True if an error occurred
        '''
        # Creates a directory in which to save data if necessary
        script_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(os.path.dirname(script_path[:-1]), 'data')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        # Saves files
        self.test_df.to_csv(os.path.join(data_dir, 'test_df.csv'))
        var_to_pickle(self.gbm, os.path.join(data_dir, 'model_gbm.pk'))
        var_to_pickle(self.lrm, os.path.join(data_dir, 'model_lrm.pk'))
        return None


if __name__ == '__main__':
    bm = build_models()
