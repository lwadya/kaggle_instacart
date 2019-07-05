import numpy as np
import copy
import itertools

class cross_val_by_group:
    '''
    Splits a dataframe into folds for cross validation while keeping groups of
    rows (based on common column value) in the same folds. Also allows for
    simple fitting and predicting using sklearn models from within the class
    structure.
    '''

    def __init__(self, df, feature_cols, target_col, group_col, num_folds=5,
                 seed=None):
        '''
        Designates fold splits of the dataframe

        Args:
            df (dataframe): dataframe housing both training and validation data
            feature_cols (string list): dataframe feature column names
            target_col (string): dataframe target column name
            group_col (string): dataframe column to not be split up across folds
            num_folds (int): number of folds to use for cross validation
            seed (int): random seed, can also be None to ignore random seed

        Returns:
            None
        '''
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.num_folds = num_folds
        if seed:
            rndm = np.random.RandomState(seed)
        else:
            rndm = np.random

        # Randomly splits group members into folds
        members = self.df[group_col].unique()
        rndm.shuffle(members)
        group_folds = np.split(members, self.num_folds)

        # Compile list of validation indices for each fold
        self.val_indices = [set(self.df[self.df[group_col].isin(g)].index.
                            values.tolist()) for g in group_folds]

        # Compile list of train indices for each fold
        all_indices = set(self.df.index.values.tolist())
        self.train_indices = [all_indices - i for i in self.val_indices]

    def fit(self, model):
        '''
        Fits the given model for the designated number of folds and assigns all
        models to a class variable

        Args:
            model (sklearn model object): can be any model that follows sklearn
                                          fit, score, predict paradigm

        Returns:
            None
        '''
        self.models = []
        if 'xgboost.sklearn.XGBClassifier' in str(type(model)):
            xgb = True
        else:
            xgb = False

        for it, iv in zip(self.train_indices, self.val_indices):
            X_t = self.df.loc[it, self.feature_cols]
            y_t = self.df.loc[it, self.target_col]
            X_v = self.df.loc[iv, self.feature_cols]
            y_v = self.df.loc[iv, self.target_col]
            self.models.append(copy.deepcopy(model))

            if xgb:
                self.models[-1].fit(X_t, y_t,
                                    eval_set=[(X_t, y_t), (X_v, y_v)],
                                    eval_metric='auc',
                                    early_stopping_rounds=20,
                                    verbose=False)
            else:
                self.models[-1].fit(X_t, y_t)

    def predict(self, threshold=.5):
        '''
        Generates predictions for each fold using the respective model and
        validation set

        Args:
            threshold (float): if not .5, uses predict_proba to get threshold-
                               adjusted predictions

        Returns:
            list: numpy arrays of predicted values for each fold in a list
        '''
        pred = []

        if threshold == .5:
            for model, idx in zip(self.models, self.val_indices):
                pred.append(model.predict(self.df.loc[idx, self.feature_cols]))
        else:
            for model, idx in zip(self.models, self.val_indices):
                probs = model.predict_proba(self.df.loc[idx, self.feature_cols])
                pred.append((probs[:, 1] >= threshold).astype(int))

        return pred

    def get_actual(self):
        '''
        Retrieves actual target values separated by fold

        Args:
            None

        Returns:
            list: numpy arrays of actual values for each fold in a list
        '''
        return [self.df.loc[idx, self.target_col] for idx in self.val_indices]

    def get_df_folds(self, limit_cols=True):
        '''
        Retrieves dataframe separated by folds

        Args:
            limit_cols (bool): if True, returns only features and target columns

        Returns:
            list: pandas dataframe for each fold in a list
        '''
        return [self.df.loc[idx, self.feature_cols + [self.target_col]]
                for idx in self.val_indices]

    def score(self, scorer, threshold=.5):
        '''
        Generates a score of actual vs predicted target values for each fold
        using designated scorer

        Args:
            scorer (sklearn score object): any object that provides a score
                                           based on actual and predicted value
                                           arrays
            threshold (float): if not .5, uses threshold-adjusted predictions

        Returns:
            list: a score for each fold
        '''
        preds = self.predict(threshold)
        actuals = self.get_actual()
        scores = [scorer(a, p) for a, p in zip(actuals, preds)]
        return np.mean(scores)


def train_test_by_group(df, group_col, test_size=.1, seed=None):
    '''
    Splits a dataframe into train and test partitions while preventing the
    designated group from being separated between sets

    Args:
        df (dataframe): dataframe housing both training and test data
        group_col (string): dataframe column to not be split up across sets
        test_size (float): proportion of dataframe to use for test holdout
        seed (int): random seed, can also be None to ignore random seed

    Returns:
        dataframes: training data followed by test data
    '''
    if seed:
        rndm = np.random.RandomState(seed)
    else:
        rndm = np.random

    # Splits dataframe into test and train dataframes
    members = df[group_col].unique()
    test_members = rndm.choice(members,
                               size=int(members.size * test_size),
                               replace=False)
    test_df = df[df[group_col].isin(test_members)]
    train_df = df[~df[group_col].isin(test_members)]

    return train_df, test_df
