# stacking method


import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge , HuberRegressor # 第二层的模型一般为了防止过拟合会采用简单的模型
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold  # 就是做多次Kfold CV


class Stacking(object):
    # Stacking原来是这么做的 不知道theoretical guarantee是什么 但是大家都是这么做的
    def __init__(self, n_fold=10, stack_model='Ridge'):
        self.n_fold = n_fold
        self.stack_model = stack_model

    def get_stacking(self, oof_list, prediction_list, labels):
        '''
        :param oof_list: out-of-fold predictions
        :param prediction_list: test predictions
        :param labels: true labels of the training data set
        :return: stacking oof predictions of the training set and the testing set
        '''
        train_stack = np.vstack(oof_list).transpose()   # vertical stack起来
        test_stack = np.vstack(prediction_list).transpose()

        repeats = len(oof_list)  # 第一层做了多少个模型第二层就重复多少次CV  但也可以自己定啦
        kfolder = RepeatedKFold(n_splits=self.n_fold, n_repeats=repeats, random_state=4590)
        kfold = kfolder.split(train_stack, labels)   # stacking的这些模型里面也要做CV的
        preds_list = list()  # predictions of the testing data's labels
        stacking_oof = np.zeros(train_stack.shape[0])  # predictions of the oof training data's labels

        for train_index, vali_index in kfold:
            k_x_train = train_stack[train_index]
            k_y_train = labels.loc[train_index]
            k_x_vali = train_stack[vali_index]

            assert self.stack_model in ['Ridge', 'Huber']
            if self.stack_model=='Ridge':
                stacking_model = BayesianRidge() # BayesianRidge
            if self.stack_model=='Huber':
                stacking_model = HuberRegressor()
            stacking_model.fit(k_x_train, k_y_train)

            k_pred = stacking_model.predict(k_x_vali)
            stacking_oof[vali_index] = k_pred

            preds = stacking_model.predict(test_stack)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(labels, stacking_oof)
        print(f'stacking fold mae training error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold * repeats)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        stacking_prediction = list(preds_df.mean(axis=1))

        return stacking_oof, stacking_prediction
