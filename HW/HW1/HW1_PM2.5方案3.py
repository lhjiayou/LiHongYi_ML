# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:10:20 2022

@author: 18721
"""

#-*- coding:utf-8 -*-
# @File    : PM2.5Prediction.py
# @Date    : 2019-05-19
# @Author  : 追风者
# @Software: PyCharm
# @Python Version: python 3.6

'''
利用线性回归Linear Regression模型预测 PM2.5

特征工程中的特征选择与数据可视化的直观分析
通过选择的特征进一步建立回归模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''数据读取与预处理'''
# DataFrame类型
train_data = pd.read_csv("./Dataset/train.csv")
train_data.drop(['Date', 'stations', 'observation'], axis=1, inplace=True)
# train_data.shape (4320, 24)    20*12*18=4320

ItemNum=18
#训练样本features集合
X_Train=[]
#训练样本目标PM2.5集合
Y_Train=[]

for i in range(int(len(train_data)/ItemNum)):   #240，因为12个月，每个月20天
    observation_data = train_data[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
    for j in range(15):  #也是按照每一天进行rolling的
        x = observation_data.iloc[:, j:j + 9]  #18*9
        y = int(observation_data.iloc[9,j+9])  #第一个9是因为，pm2.5在18中物质中的编号(0-17)是9
        # 将样本分别存入X_Train、Y_Train中
        X_Train.append(x)
        Y_Train.append(y)
# print(X_Train)   240*15=3600个数据，每个数据是18*9
# print(Y_Train)   240*15=3600个数据



# 选择最具代表性的特征：PM10、PM2.5、SO2
data = pd.read_csv("./Dataset/train.csv")
wuzhi=data['observation'].unique()
dic={}
for i in wuzhi:
    dic[i]=data[data['observation']==i].drop(['Date', 'stations', 'observation'], axis=1).values.reshape(-1,)
dic=pd.DataFrame(dic)  
dic.corr()  
# dic.dtypes 全部是object
dic.drop(['RAINFALL'],axis=1,inplace=True)
dic=dic.astype('float')
dic.corr() ['PM2.5'].sort_values(ascending=False)
# PM2.5         1.000000
# PM10          0.776426
# NO2           0.449113
# NOx           0.375564
# SO2           0.370831
# O3            0.356670
# THC           0.352159
# NMHC          0.291778
# CO            0.283119
# CH4           0.254657
# WD_HR         0.186138
# WIND_DIREC    0.156990
# NO            0.029970
# AMB_TEMP     -0.017127
# WS_HR        -0.045458
# WIND_SPEED   -0.084703
# RH           -0.264196


# 然后下面就是pm2.5prediction的内容，好像很复杂，其实可以使用hands书本上的内容来学习