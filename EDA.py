# Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
train_data, test_data = pd.read_csv('./train_dataset/train_dataset.csv'), \
                        pd.read_csv('./test_dataset/test_dataset.csv')

original_columns = ['用户编码', '用户实名制是否通过核实', '用户年龄', '是否大学生客户', '是否黑名单客户', '是否4G不健康客户',
       '用户网龄（月）', '用户最近一次缴费距今时长（月）', '缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）',
       '用户账单当月总费用（元）', '用户当月账户余额（元）', '缴费用户当前是否欠费缴费', '用户话费敏感度', '当月通话交往圈人数',
       '是否经常逛商场的人', '近三个月月均商场出现次数', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店', '当月是否看电影',
       '当月是否景点游览', '当月是否体育场馆消费', '当月网购类应用使用次数', '当月物流快递类应用使用次数',
       '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
       '当月旅游资讯类应用使用次数', '信用分']

train_data.columns = ['id', 'is_real_name', 'age', 'is_college_student', 'is_blacklist', 'is_illbeing_4g',
                        'surfing_time', 'last_pay_month', 'last_pay_account', 'avg_pay_account',
                        'this_month_account',
                        'this_month_balance', 'is_arrearage', 'account_sensitivity', 'this_month_call_num',
                        'is_shopping', 'avg_shopping_num', 'is_wanda', 'is_sam', 'is_movie', 'is_travel', 'is_sports',
                        'online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                        'train_num', 'travel_num', 'score']

test_data.columns = ['id', 'is_real_name', 'age', 'is_college_student', 'is_blacklist', 'is_illbeing_4g',
                        'surfing_time', 'last_pay_month', 'last_pay_account', 'avg_pay_account',
                        'this_month_account',
                        'this_month_balance', 'is_arrearage', 'account_sensitivity', 'this_month_call_num',
                        'is_shopping', 'avg_shopping_num', 'is_wanda', 'is_sam', 'is_movie', 'is_travel', 'is_sports',
                        'online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                        'train_num', 'travel_num']

# explore the characteristics of different features
# First look the distribution of score
sns.distplot(train_data['score'], bins=50, kde=False, color="darkblue")
plt.show()
# How to overcome the imbalance in the prediction problem?
# 没有解决这个问题 可以试一试data augmentation

################### Personal Data #####################
# score & is_real_name
sns.boxplot(x="is_real_name", y="score", palette="Greens",
            data=train_data, linewidth=4)
plt.show()

# score & is_college_student
sns.boxplot(x="is_college_student", y="score", palette="Blues",
            data=train_data, linewidth=4)
plt.show()

# score & is_blacklist
sns.boxplot(x="is_blacklist", y="score", palette="Greys",
            data=train_data, linewidth=4)
plt.show()
# Contradict with our common sense. what happened?
# This is because the strong imbalance of these two categories. Strong bias exists.
# 这里没有处理欸
sum(train_data['is_blacklist']==1)
sum(train_data['is_blacklist']==0)

# score & is_illbeing_4g
sns.boxplot(x="is_illbeing_4g", y="score", palette="Oranges",
            data=train_data, linewidth=4)
plt.show()

# score & surfing_time
sns.scatterplot(
    data=train_data,
    x="surfing_time", y="score",
    s=10)
plt.show()

# score & age
sns.scatterplot(
    data=train_data,
    x="age", y="score",
    s=10, color = "purple")
plt.show()
# 0 and >100 are abnormal values. We should delete them.
# 这里处理了

# heatmap
features_personal  = ['score','is_real_name', 'is_college_student', \
       'is_blacklist', 'is_illbeing_4g', 'age','surfing_time']
corr_personal = train_data[features_personal].corr()
heatmap_personal = sns.heatmap(corr_personal, annot=True,cmap="Greens", linewidths = 1)
heatmap_personal.set_xticklabels(heatmap_personal.get_xticklabels(), rotation=360)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()
# Strong correlation features: surfing_time age

################# Consumption Data ####################
### Below are mobile phone payment info
# score & last_pay_month
# The values are 0/1?
sns.boxplot(x="last_pay_month", y="score", palette="Purples",
            data=train_data, linewidth=4)
plt.show()

# score & last_pay_account
# categorical data mixed
sns.scatterplot(
    data=train_data,
    x="last_pay_account", y="score",
    s=10, color = "brown")
plt.show()
pd.set_option('display.max_rows', None)
pd.DataFrame(train_data['last_pay_account'].value_counts(ascending=True))

# there are several types of charge amounts, and we may encode them

# score & avg_pay_account, this_month_account, this_month_balance, this_month_call_num
sns.pairplot(train_data[['score','last_pay_account','avg_pay_account',\
                         'this_month_account','this_month_balance', 'this_month_call_num']])
plt.show()

# score & is_arrearage
sns.boxplot(x="is_arrearage", y="score", palette="Reds",
            data=train_data, linewidth=4)
plt.show()
sum(train_data['is_arrearage']==1)
# strong bias exist due to imbalance

# score & account_sensitivity
# 用户话费敏感度一级表示敏感等级最大
sns.boxplot(x="account_sensitivity", y="score", palette="rainbow",
            data=train_data, linewidth=4)
plt.show()
# The 0 values are missing values
sum(train_data['account_sensitivity']==0)

### Below are life consume behaviour
# score & is_shopping, is_wanda, is_sam, is_movie, is_travel, is_sports
features_life_consume  = ['score','is_shopping', 'is_wanda', \
       'is_sam', 'is_movie', 'is_travel','is_sports']
corr_life_consume = train_data[features_life_consume].corr()
heatmap_life_consume = sns.heatmap(corr_life_consume, annot=True,cmap="Reds", linewidths = 1)
heatmap_life_consume.set_xticklabels(heatmap_life_consume.get_xticklabels(), rotation=360)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# score & avg_shopping_num
# no trend. omit

##################### App Data ########################

# score & app using info
features_app  = ['score','online_shopping_num','logistics_num', 'financing_num', \
       'video_num', 'airplant_num', 'train_num','travel_num']
corr_app = train_data[features_app].corr()
heatmap_app = sns.heatmap(corr_app, annot=True,cmap="Blues", linewidths = 1)
heatmap_app.set_xticklabels(heatmap_app.get_xticklabels(), rotation=360)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()

# there exists some abnormal data
# 羊毛党
plt.subplot(221)
sns.scatterplot(
    data=train_data,
    x="online_shopping_num", y="score",
    s=10, color = "brown")
plt.subplot(222)
sns.scatterplot(
    data=train_data,
    x="logistics_num", y="score",
    s=10, color = "brown")
plt.subplot(223)
sns.scatterplot(
    data=train_data,
    x="financing_num", y="score",
    s=10, color = "brown")
plt.subplot(224)
sns.scatterplot(
    data=train_data,
    x="video_num", y="score",
    s=10, color = "brown")
plt.show()

plt.subplot(221)
sns.scatterplot(
    data=train_data,
    x="airplant_num", y="score",
    s=10, color = "brown")
plt.subplot(222)
sns.scatterplot(
    data=train_data,
    x="train_num", y="score",
    s=10, color = "brown")
plt.subplot(223)
sns.scatterplot(
    data=train_data,
    x="travel_num", y="score",
    s=10, color = "brown")
plt.show()
