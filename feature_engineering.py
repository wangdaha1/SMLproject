# feature engineering

import pandas as pd
import os
import sys
import numpy as np
from collections import Counter
from utils import timer
from selector import Selector
import warnings

warnings.filterwarnings(action='ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Processing(object):
    def __init__(self, selector=False, ascending=False):
        self.selector = selector
        self.ascending = ascending

    @ staticmethod
    def _get_columns_name():
        columns_name = ['id', 'is_real_name', 'age', 'is_college_student', 'is_blacklist', 'is_illbeing_4g',
                        'surfing_time', 'last_pay_month', 'last_pay_account', 'avg_pay_account','this_month_account',
                        'this_month_balance', 'is_arrearage', 'account_sensitivity', 'this_month_call_num',
                        'is_shopping', 'avg_shopping_num', 'is_wanda', 'is_sam', 'is_movie', 'is_travel', 'is_sports',
                        'online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                        'train_num', 'travel_num', 'score']
        return columns_name

    def _get_data_plusaugmentation(self):
        columns_name = self._get_columns_name()  # 如果希望在方法裡面调用静态类，那么把方法定义成类方法是合适的

        train_data, test_data = pd.read_csv('./train_dataset/train_dataset.csv'), \
                                pd.read_csv('./test_dataset/test_dataset.csv')

        # try data augmentation

        test_data['score'] = -1
        train_data.columns = columns_name
        test_data.columns = columns_name

        dataset = pd.concat([train_data, test_data], ignore_index=True)
        return dataset



    @staticmethod
    def _get_age_type(item):
        if item==0:
            return -1 # missing data
        if item <15 and item >0:
            return 'Child'
        if item >=15 and item <30:
            return 'Young'
        if item >=30 and item <50:
            return 'Middle'
        if item>=50:
            return 'Old'

    @staticmethod
    def _get_boolean_columns(dataset):
        dataset = dataset.copy()
        boolean_columns = list()

        for column in dataset.columns:
            nunique = dataset[column].nunique() # number of categories
            if nunique == 2:
                boolean_columns.append(column)

        return boolean_columns

    @staticmethod
    def _get_missing_value(item):
        if item == 0:
            return 1
        return 0

    @staticmethod
    def _get_abnormal_label(item):
        '''
        for the app usage data. We devide them into several groups.
        '''
        if item == 0:
            return 0
        if item < 10:
            return 1
        if item < 100:
            return 2
        if item < 1000:
            return 3
        else:
            return 4

    @staticmethod
    def _recombine_boolean_columns(dataset, boolean_columns):
        # Boolean型特征二次组合，将bool类型重新组合 也就是交互特征啦 比如可能是否大学生+是否黑名单时才是强特
        # 可是这个写法还蛮奇怪的 难道没有直接的函数可以写出is_blacklist:is_college这种形式的交互特征了吗
        # 通过2进制编码
        dataset = dataset.copy()

        bin_base = 1
        dataset['boolean_bin'] = 0  # 初始化
        for column in boolean_columns:
            dataset['boolean_bin'] += dataset[column] * bin_base
            bin_base = 2 * bin_base

        # 由于有些编码情况过于少，应该合并为-1，尝试了不合并，小于2次5次和10次的合并，线下最好的为小于5合并
        counter = Counter(dataset['boolean_bin'])
        counter_dict = dict()
        for item, count in counter.items():
            if count < 5:
                counter_dict[item] = -1
            else:
                counter_dict[item] = count
        dataset['boolean_bin'] = dataset['boolean_bin'].map(counter_dict)

        # One-Hot
        dataset = pd.get_dummies(dataset, columns=['boolean_bin'])
        return dataset

    @staticmethod
    def _get_recharge_way(item):
        # 通过最后充值金额判断充值方式 通过充值金额分三类，充值金额=0，充值金额能被10整除，充值金额不能被10整除
        # 可是在原始数据里面 好像不是以10为一个单位的额 好像是0.998折
        taocan = [30, 499, 100, 299.4, 50, 19.96, 9.98, 199.6, 29.94, 49.9, 99.8] # 11种套餐金额

        if item == 0:  # 这里后面可以看到 是把他认为是missing data了  这里就把missing data处理成-1了  但是其实也不是很合理吧
            return -1
        for i in range(len(taocan)): # 套餐里面的几种金额
            if item == taocan[i]:
                return taocan[i]
        else:   # 其他的充值金额
            return 0

    @staticmethod
    def _get_use_discount(item):
        list_yes = [499, 299.94, 19.96, 9.98, 199.6, 29.94, 49.9, 99.8]
        list_no = [30, 100, 50]
        if item in list_yes:
            return 'yes'
        if item in list_no:
            return 'no'
        if item==0: # 因为这个函数是接着上面那个_get_recharge_way的
            return 'notTC'

    @staticmethod
    def _shopping_encoder(item):
        is_shopping = item['is_shopping']
        avg_shopping_num = item['avg_shopping_num']

        if is_shopping == 0:
            if avg_shopping_num < 10:
                return 0
            elif avg_shopping_num < 20:
                return 1
            else:
                return 2
        else:
            if avg_shopping_num < 20:
                return 3
            return 4

    @staticmethod
    def _get_app_rate(dataset):
        dataset = dataset.copy()

        app_num_columns = ['online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                           'train_num', 'travel_num']
        dataset['app_sum'] = dataset[app_num_columns].apply(lambda item: np.log1p(np.sum(item)), axis=1)  # log1p(x)=log(1+x) 做一个平滑
                             # axis = 1: apply function to each row
        for column in app_num_columns:
            column_name = f'{column}_rate'  # combine
            dataset[column_name] = np.log1p(dataset[column]) / dataset['app_sum']  # 分子和分母都加了一个log1p的

        # dataset = dataset.drop(columns=['helper_sum'])
        return dataset

    @staticmethod
    def _data_encoder(dataset, num_columns):
        # 这个函数没看懂在干嘛
        dataset = dataset.copy()

        # # LabelEncoder
        # for column in num_columns:
        #     mapping_dict = dict(zip(dataset[column].unique(), range(0, dataset[column].nunique())))
        #     dataset[column] = dataset[column].map(mapping_dict)
        # qcut
        for column in num_columns:
            dataset[column] = pd.qcut(dataset[column], 20, labels=False, duplicates='drop')

        train_data = dataset[dataset['score'] > 0]
        train_data['helper'] = pd.cut(train_data['score'], 5, labels=False)  # 将score分成五个level
        train_data = pd.get_dummies(train_data, columns=['helper'])
        helper_columns = ['helper_0', 'helper_1', 'helper_2', 'helper_3', 'helper_4']

        for column in num_columns:
            for helper_column in helper_columns:
                column_name = f'{column}_{helper_column}_mean'
                column_df = train_data.groupby(by=[column])[helper_column].agg('mean').reset_index(name='mean')
                column_dict = column_df.set_index(column)['mean'].to_dict()

                dataset[column_name] = dataset[column].map(column_dict)

        return dataset

    def _get_operation_features(self, dataset):
        dataset = dataset.copy()

        # 年纪类型
        dataset['age_type'] = dataset['age'].apply(self._get_age_type)
        dataset = pd.get_dummies(dataset, columns=['age_type'])

        # 充值方式 是否使用折扣
        dataset['recharge_way'] = dataset['last_pay_account'].apply(self._get_recharge_way)
        dataset['use_discount'] = dataset['recharge_way'].apply(self._get_use_discount)
        dataset = pd.get_dummies(dataset, columns=['recharge_way','use_discount']) # 进行onehot编码

        # 稳定性
        # 当月话费 / (近6个月平均话费 + 5)
        dataset['month_half_year_stable'] = dataset['this_month_account'] / (dataset['avg_pay_account'] + 5)
        dataset['month_half_year_diff'] = dataset['this_month_account'] - dataset['avg_pay_account']
        # 当月话费 / (当月账户余额 + 5)
        dataset['use_left_stable'] = dataset['this_month_account'] / (dataset['this_month_balance'] + 5)
        dataset['use_left_diff'] = dataset['this_month_account'] - dataset['this_month_balance']

        # 商场行为编码
        dataset['shopping_encoder'] = dataset[['is_shopping', 'avg_shopping_num']].apply(self._shopping_encoder, axis=1)
        dataset = pd.get_dummies(dataset, columns=['shopping_encoder'])

        # 上网时长
        dataset['surfing_time_copy'] = dataset['surfing_time']
        dataset['surfing_time_copy'] = pd.qcut(dataset['surfing_time_copy'], 5, labels=False) # qcut是让分组里的每个数据量都大致相同
        dataset = pd.get_dummies(dataset, columns=['surfing_time_copy'])

        # APP打开占比
        dataset = self._get_app_rate(dataset)

        return dataset

    @timer(func_name='Processing.get_processing')  # 这个函数是自己定义的哈  不是class里面自带的
    def get_processing(self):
        # 最终处理数据生成特征的函数
        dataset = self._get_data_plusaugmentation()

        boolean_columns = self._get_boolean_columns(dataset)
        remove_columns = ['id', 'score']
        num_columns = list()  # 摘取其余的特征
        for column in dataset.columns:
            if column in remove_columns:
                continue
            if column in boolean_columns:
                continue
            num_columns.append(column)

        # 异常字段处理：手动分箱
        abnormal_columns = ['online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                            'train_num', 'travel_num']
        abnormal_encoder_columns = list()
        for column in abnormal_columns:
            encoder_column = f'{column}_encoder'
            dataset[encoder_column] = dataset[column].apply(self._get_abnormal_label)
            abnormal_encoder_columns.append(encoder_column)
        dataset = pd.get_dummies(dataset, columns=abnormal_encoder_columns)
        #_get_abnormal_label函数只是把数据变成1234  但是要通过get_dummies变成哑变量0001

        # 缺失值单独抽离特征：无效  这怎么可能有效啊！无语子
        # 那既然这样无效的话 用什么办法处理缺失值呢？
        # 是认为 除了boolean特征和刚刚处理过的app使用次数特征之外 其余的0值就是missing data啦
        # for column in num_columns:
        #     # abnormal已处理过，continue
        #     if column in abnormal_columns:
        #         continue
        #     column_name = f'{column}_missing'
        #     dataset[column_name] = dataset[column].apply(self._get_missing_value)

        # 将bool类型重新组合 也就是交互特征啦
        # 生成的特征有点过多了 可以考虑去掉
        # dataset = self._recombine_boolean_columns(dataset, boolean_columns)

        # ??看看是啥特征 没看懂是在干嘛
        # dataset = self._data_encoder(dataset, ['surfing_time', 'age'])

        # 业务逻辑特征
        dataset = self._get_operation_features(dataset)

        if self.selector:
            train_data = dataset[dataset['score'] > 0]
            y_data = train_data['score']
            x_data = train_data.drop(columns=['id', 'score'])
            # 选特征
            select_features = Selector(ascending=self.ascending).get_select_features(x_data, y_data)
            select_features.extend(['id', 'score'])
            dataset = dataset[select_features]

        return dataset



def processing_main(selector=False, ascending=False):
    processing = Processing(selector=selector, ascending=ascending)
    dt = processing.get_processing()

    features_name = os.path.join(BASE_DIR,'results', 'features.csv')
    dt.to_csv(features_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    processing_main(selector=False, ascending=False)


