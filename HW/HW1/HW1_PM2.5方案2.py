# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:28:50 2022

@author: 18721
"""

#-*- coding:utf-8 -*-
# @File    : Predict_PM2dot5.py
# @Date    : 2019-05-19
# @Author  : 追风者
# @Software: PyCharm
# @Python Version: python 3.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据读取与预处理
train_data = pd.read_csv("./Dataset/train.csv")  
#4320*27，为什么是4320，因为每个月前20天，总共12个月，因此就有240天数据，每天18种物质，就是4320
train_data.drop(['Date', 'stations'], axis=1, inplace=True)  ##4320*25

# train_data=train_data[train_data['observation']=='PM2.5'] #现在只有240行，因为有18种物质
# train_data.shape  (240, 25)
# 除去第一个observation列之后其实是24列，按照rolling=9的滑动，也就是前9个预测后一个，可以形成15个样本
# 240*15 将是3600个样本，这也是方案1提供的

column = train_data['observation'].unique() #就是18种，要这么干嘛？
# print(column)

#下面就是对数据进行改造
new_train_data = pd.DataFrame(np.zeros([24*240, 18]), columns=column)   #5760*18

for i in column:
    train_data1 = train_data[train_data['observation'] == i]
    # Be careful with the inplace, as it destroys any data that is dropped!
    train_data1.drop(['observation'], axis=1, inplace=True)
    train_data1 = np.array(train_data1)
    train_data1[train_data1 == 'NR'] = '0'
    train_data1 = train_data1.astype('float')
    train_data1 = train_data1.reshape(1, 5760)
    train_data1 = train_data1.T
    new_train_data[i] = train_data1

label = np.array(new_train_data['PM2.5'][9:], dtype='flo这儿at32')  #因为要用前面9个预测后面的，这个是5751

# 探索性数据分析 EDA
# 最简单粗暴的方式就是根据 HeatMap 热力图分析各个指标之间的关联性
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(new_train_data.corr(), fmt="d", linewidths=0.5, ax=ax) 
#new_train_data.corr()生成18*18的相关系数矩阵
plt.show()



# 模型选择
# a.数据归一化
# 使用前九个小时的 PM2.5 来预测第十个小时的 PM2.5，使用线性回归模型
PM = new_train_data['PM2.5']
PM_mean = int(PM.mean())   #均值
PM_theta = int(PM.var()**0.5)   #标准差
PM = (PM - PM_mean) / PM_theta  #z-score
w = np.random.rand(1, 10)   #首先预定义w，size是(1,10)
theta = 0.1  #可能是学习率
m = len(label)  #样本数目
for i in range(100):  #这个100应该是iteration次数
    #不同于方案1，是按照(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon)作为收敛判别条件
    #方案2中就是直接给定迭代的次数，而且方案1封装的很好
    loss = 0
    i += 1
    gradient = 0
    
    for j in range(m):   #遍历每个样本
        x = np.array(PM[j : j + 9])
        x = np.insert(x, 0, 1)  #在0的位置插入1，给b的位置
        error = label[j] - np.matmul(w, x) 
        loss += error**2
        gradient += error * x
    #这个学习仍然是梯度下降，不是SGD，是在所有的sample全部学习完成之后才算的一次epoch的loss和w的更新

    loss = loss/(2*m)
    print(loss)
    w = w+theta*gradient/m  #这儿是加，因为上面gradient += error * x，其实是应该error * -x
# 上面100epoch训练完成之后的w
w
# array([[21.03899803,  0.18970145, -0.05888972,  0.03859881,  0.10942321,
#          0.32802808,  1.21989149, -3.51326613,  2.58627368, 13.62450518]])
intercept_=w[0,0]  #21.038998026849757
coef_ =w[0,1:]
# array([ 0.18970145, -0.05888972,  0.03859881,  0.10942321,  0.32802808,
#         1.21989149, -3.51326613,  2.58627368, 13.62450518])



