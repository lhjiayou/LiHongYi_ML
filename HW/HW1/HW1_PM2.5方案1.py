# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 21:15:23 2022

@author: 18721
"""


'''
只使用了pm2.5的线性模型,并没有划分什么验证集，也没有使用交叉验证这些，反正线性模型没有任何需要调整的超参数
利用 Linear Regression 线性回归预测 PM2.5 
该方法参考黑桃大哥的优秀作业-|vv|-
'''

# 导入必要的包 numpy、pandas以及scikit-learn归一化预处理
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 指定相对路径
path = "./Dataset/"

# 利用 pandas 进行读取文件操作
train = pd.read_csv(path + 'train.csv', engine='python', encoding='utf-8')  
# train.shape  (4320, 27)
test = pd.read_csv(path + 'test.csv', engine='python', encoding='gbk',header=None)
# test.shape  (4319, 11)但是第一行被错误读成了列名，需要加上header=none

train = train[train['observation'] == 'PM2.5']   
# train.shape  (240, 27),因为4320/18=240

# test = test[test['AMB_TEMP'] == 'PM2.5']  这个其实是不对的
test=test[test.iloc[:,1]== 'PM2.5']
# test.shape   (240, 11)

# 删除无关特征
# train = train.drop(['Date', 'stations', 'observation'], axis=1)
train.drop(['Date', 'stations', 'observation'], axis=1,inplace=True)  #原地更改就可以
test_x = test.iloc[:, 2:]   #第一列是id，第二列是pm2.5，不需要写

#下面的代码其实就是显式地创建训练x与y，但其实这样会很占内存
#就是将240*24变成了15个240*9，滑动窗口得到的数据
train_x = []
train_y = []
for i in range(15):
    x = train.iloc[:, i:i + 9]
    # notice if we don't set columns name, it will have different columns name in each iteration
    x.columns = np.array(range(9))
    y = train.iloc[:, i + 9]
    y.columns = np.array(range(1))
    train_x.append(x)
    train_y.append(y)

# review "Python for Data Analysis" concat操作
# train_x and train_y are the type of Dataframe
# 取出 PM2.5 的数据，训练集中一共有 240 天，每天取出 15 组 含有 9 个特征 和 1 个标签的数据，共有 240*15*9个数据
train_x = pd.concat(train_x) # (3600, 9) Dataframe类型
train_y = pd.concat(train_y) #(3600,)

# type(train_x.iloc[0,0])  现在是str

# 将str数据类型转换为 numpy的 ndarray 类型，或者直接astype好像就可以
# train_y = np.array(train_y, float)
# test_x = np.array(test_x, float)
# print(train_x.shape, train_y.shape)
train_x=train_x.astype('float')
train_y=train_y.astype('float')


# 进行标准缩放，即数据归一化
ss = StandardScaler()  #这个应该不是归一化而是z_score标准化
# 进行数据拟合
ss.fit(train_x)
# 进行数据转换
train_x = ss.transform(train_x)
# ss.fit(test_x)  不能写
test_x = ss.transform(test_x)

# #上面的 ss.fit(test_x)应该石油问题的
# ss = StandardScaler()
# ss.fit(train_x)
# ss.mean_
# # array([19.48388889, 19.99333333, 20.59805556, 21.23916667, 21.87944444,
# #        22.49472222, 23.03611111, 23.51055556, 23.8525    ])
# ss.var_
# # array([251.50529599, 259.2694    , 266.139274  , 274.55474375,
# #        280.50379969, 287.40330548, 292.47814043, 296.56377747,
# #        297.54963264])
# test_x1 = ss.transform(test_x)

# ss.fit(test_x)
# ss.mean_
# # array([27.025     , 26.85833333, 27.125     , 27.27083333, 27.6625    ,
# #        27.7375    , 27.95833333, 28.3125    , 28.44166667])
# ss.var_
# # array([481.399375  , 460.83826389, 471.759375  , 438.91414931,
# #        472.94026042, 447.71026042, 462.13159722, 447.33984375,
# #        458.77993056])
# test_x2 = ss.transform(test_x)


# 定义评估函数
# 计算均方误差（Mean Squared Error，MSE）
# r^2 用于度量因变量的变异中 可以由自变量解释部分所占的比例 取值一般为 0~1
def r2_score(y_true, y_predict):
    # 计算y_true和y_predict之间的MSE
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    # 计算y_true和y_predict之间的R Square
    return 1 - MSE / np.var(y_true)

# 线性回归
class LinearRegression:
    '''为什么需要自定义线性回归器这个class'''
    def __init__(self):
        # 初始化 Linear Regression 模型
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        '''方式1，求取解析解，closed_form result'''
        # 根据训练数据集X_train, y_train训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 对训练数据集添加 bias
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  #就是在x数据前面加上1
        #下面是按照解析解求得，这个在hands书的第三章好像有写
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):  
        '''方式2，用gd求取'''
        #写在函数里面，可以lr.fit_gd(输入参数)，如果将eta=0.01, n_iters=1e4写在上面的__init__中，那就是
        #实例化的时候写，也就是lr=LinearRegression(eta=0.01, n_iters=1e4)
        '''
        :param X_train: 训练集
        :param y_train: label
        :param eta: 学习率
        :param n_iters: 迭代次数
        :return: theta 模型参数
        '''
        # 根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 定义损失函数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
            
        # 对损失函数求导，就是对上面的J写出了人工的求导
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)  
            #这儿是X_b.dot(theta) - y 

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            #这儿写参数默认值其实没什么大的作用，不必是这个函数的内部参数，直接使用
            #调用fit_gd函数时候输入的参数就可以，对fit_gd而言是内部参数，对于gradient_descent
            #而言其实是外部参数
            '''
            :param X_b: 输入特征向量
            :param y: lebel
            :param initial_theta: 初始参数
            :param eta: 步长
            :param n_iters: 迭代次数
            :param epsilon: 容忍度
            :return:theta：模型参数
            '''
            theta = initial_theta   #初始的模型参数
            cur_iter = 0

            while cur_iter < n_iters:  #不断迭代
                gradient = dJ(theta, X_b, y)   #计算梯度
                last_theta = theta             #上一次的参数w
                theta = theta - eta * gradient #更新之后的参数w
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):  
                    #小于收敛条件的时候，两次损失很接近的时候
                    break
                cur_iter += 1
            return theta   #gradient_descent函数返回的是梯度下降训练完成后的参数w

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1]) # 初始化theta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        #这儿的eta, n_iters其实是上面的fit_gd函数输入的eta=0.01, n_iters=1e4
        #其实是在fig_gd函数中调用上面的gradient_descent函数
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self   
        #fit_gd函数返回的是整个lr本身

    def predict(self, X_predict):
        # 给定待预测数据集X_predict，返回表示X_predict的结果向量，需要上面fit_gd先完成
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        #必须先训练再预测
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        #必须让w的长度和待预测的x的长度相等
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        #每个x都是需要在前面加上1的，这个1对应的是b
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        # 根据测试数据集 X_test 和 y_test 确定当前模型的准确度
        y_predict = self.predict(X_test)  #直接调用上面的predict函数
        return r2_score(y_test, y_predict)

    def __repr__(self):   #这个什么？
        return


# 模型训练
LR = LinearRegression().fit_gd(train_x, train_y)
LR.intercept_  #24.05694444444436
LR.coef_  
# array([ 0.10915232, -0.71902954,  3.22636836, -3.43426218, -0.70907929,
#         7.80665585, -9.28257051,  0.28134532, 18.56971041])

#如果我们使用解析解，会是怎样？存为轻微差异
LR1 = LinearRegression().fit_normal(train_x, train_y)
LR1.intercept_  #24.056944444444447
LR1.coef_ 
# array([ 0.11562505, -0.7412229 ,  3.25324252, -3.43948357, -0.734357  ,
#         7.83829912, -9.29139403,  0.26732565, 18.58073518])


# 评分，可见线性模型的分数都差不多
LR.score(train_x, train_y)  #0.854900184173954
LR1.score(train_x, train_y) #0.8549004706037486

# 预测
result = LR.predict(test_x)

# 结果保存
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', engine='python', encoding='gbk')
sampleSubmission['value'] = result
sampleSubmission.to_csv( 'result方案1.csv')
