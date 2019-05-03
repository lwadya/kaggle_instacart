import numpy as np
import copy
import itertools

class cross_val_by_group:

    def __init__(self, df, feature_cols, target_col, group_col, num_folds=5,
                 seed=None):
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
        self.models = []
        for idx in self.train_indices:
            self.models.append(copy.deepcopy(model))
            self.models[-1].fit(self.df.loc[idx, self.feature_cols],
                                self.df.loc[idx, self.target_col])

    def predict(self):
        pred = []
        for model, idx in zip(self.models, self.val_indices):
            pred.append(model.predict(self.df.loc[idx, self.feature_cols]))
        return pred

    def get_actual(self):
        return [self.df.loc[idx, self.target_col] for idx in self.val_indices]

    def score(self, scorer):
        preds = self.predict()
        actuals = self.get_actual()
        scores = [scorer(a, p) for a, p in zip(actuals, preds)]
        return np.mean(scores)


def train_test_by_group(df, group_col, test_size=.1, seed=None):
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
