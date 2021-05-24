import os
import time
from functools import wraps
from itertools import combinations

from scipy.optimize import minimize
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_features():
    data_name = os.path.join(BASE_DIR, 'results', 'features.csv')
    df = pd.read_csv(data_name, header=0)
    return df


def get_ensemble_score(score_df):
    score_df = score_df.copy()
    model_list = list()
    for column in score_df.columns:
        if column in ['id', 'score']:
            continue

        model_list.append(column)
        model_names = ' + '.join(model_list)

        score_df['helper'] = np.mean(score_df[model_list], axis=1)
        mae = mean_absolute_error(score_df['score'], score_df['helper'])
        mae_score = 1 / (1 + mae)

        print(f'model name: {model_names}, mae: {mae}, score: {mae_score}')


def get_combinations(arr_list):
    combinations_list = list()
    length = len(arr_list)
    for num in range(2, length):
        for bin_item in combinations(arr_list, num):
            combinations_list.append(list(bin_item))

    return combinations_list


def get_values_by_index(value_list, index_list):
    new_values = list()

    for index, value in enumerate(value_list):
        if index in index_list:
            new_values.append(value)
    return new_values


def get_score_array(dataset):
    dataset = dataset.copy()
    score_array = list()

    remove_columns = ['id', 'score']
    for column in dataset.columns:
        if column in remove_columns:
            continue
        score_array.append(np.array(dataset[column]))

    return score_array


def get_blending_score(score_array, weights):
    weight_prediction = 0.0
    for weight, prediction in zip(weights, score_array):
        weight_prediction += weight * prediction
    return weight_prediction


def timer(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            retval = func(*args, **kwargs)
            usage_time = time.time() - t0
            print(f'{func_name} usage time: {usage_time}')
            return retval

        return wrapper

    return decorator


def get_features_importance():
    dataset = get_features()

    train_data = dataset[dataset['score'] > 0]
    y_data = train_data['score']
    x_data = train_data.drop(columns=['id', 'score'])

    params = {
        'boosting_type': 'gbdt',
        'objective': 'mae',
        'n_estimators': 10000,
        'metric': 'mae',
        'learning_rate': 0.01,
        'min_child_samples': 46,
        'min_child_weight': 0.01,
        'subsample_freq': 1,
        'num_leaves': 40,
        'max_depth': 7,
        # 'subsample': 0.6,
        # 'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 5,
        'verbose': -1,
        'seed': 4590
    }
    columns = x_data.columns
    gbm = lgb.LGBMRegressor(**params)
    gbm = gbm.fit(x_data, y_data)
    feature_importance = gbm.feature_importances_
    importance_df = pd.DataFrame({'column': columns, 'score': feature_importance})
    importance_df = importance_df.sort_values(by=['score'], ascending=False).reset_index()

    for _, rows in importance_df.iterrows():
        column = rows['column']
        importance = rows['score']
        print(f'column: {column}, importance: {importance}')


class Blending(object):
    def __init__(self, train_score_df, num_round=20):
        self.train_score_df = train_score_df
        self.num_round = num_round
        self.score_columns = self._get_score_columns()
        self.score_array = self._get_score_array()

    def _get_score_columns(self):
        df = self.train_score_df.copy()

        remove_columns = ['id', 'score']
        score_columns = list()
        for column in df.columns:
            if column not in remove_columns:
                score_columns.append(column)
        return score_columns

    def _get_score_array(self):
        score_array = list()

        for column in self.score_columns:
            score_array.append(np.array(self.train_score_df[column]))

        return score_array

    def _mae_func(self, weights):
        weight_prediction = 0.0
        y_true = self.train_score_df['score'].values
        for weight, prediction in zip(weights, self.score_array):
            weight_prediction += weight * prediction

        mae_error = mean_absolute_error(y_true, weight_prediction)
        return mae_error

    def get_best_weight(self):
        score_num = len(self.score_columns)
        best_weight = None
        best_error = 9999.9
        for _ in range(self.num_round):
            weight = np.random.dirichlet(alpha=np.ones(score_num), size=1).flatten()
            bounds = [(0, 1)] * score_num

            res = minimize(self._mae_func, weight, method='L-BFGS-B', bounds=bounds,
                           options={'disp': False, 'maxiter': 100000})
            res_error = res['fun']

            if res_error < best_error:
                best_error = res_error
                best_weight = res['x']

        # 归一化
        best_weight_sum = np.sum(best_weight)
        best_weight = best_weight / best_weight_sum

        print(f'best mae error {best_error}')
        mae_score = 1 / (1 + best_error)
        print(f'best mae score: {mae_score}')
        print(best_weight)
        return best_weight
