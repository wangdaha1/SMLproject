import pandas as pd
import os
import sys

train_data, test_data = pd.read_csv('./train_dataset/train_dataset.csv'), \
                        pd.read_csv('./test_dataset/test_dataset.csv')

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# fig_dir = os.path.join(BASE_DIR, 'plots')





class Processing(object):
    def __init__(self, selector=False, ascending=False):
        self.selector = selector
        self.ascending = ascending

    @ staticmethod
    def _get_columns_name():
        columns_name = ['id', 'is_real_name', 'age', 'is_college_student', 'is_blacklist', 'is_illbeing_4g',
                        'surfing_time_month', 'last_pay_month', 'last_pay_account', 'avg_pay_account',
                        'this_month_account',
                        'this_month_balance', 'is_arrearage', 'account_sensitivity', 'this_month_call_num',
                        'is_shopping', 'avg_shopping_num', 'is_wanda', 'is_sam', 'is_movie', 'is_travel', 'is_sports',
                        'online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                        'train_num', 'travel_num', 'score']

        return columns_name

    def _get_data(self):
        columns_name = self._get_columns_name()

        train_data_name = os.path.join(RAWDATA_PATH, 'train_dataset.csv')
        train_data = pd.read_csv(train_data_name, header=0)
        train_data.columns = columns_name

        test_data_name = os.path.join(RAWDATA_PATH, 'test_dataset.csv')
        test_data = pd.read_csv(test_data_name, header=0)
        test_data['score'] = -1
        test_data.columns = columns_name

        dataset = pd.concat([train_data, test_data], ignore_index=True)
        return dataset
