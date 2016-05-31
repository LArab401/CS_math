# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:15:26 2016

@author: Dell
"""

import numpy as np

#拟合用数据。
x=[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
y=[19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01]

#LM算法
#初始猜测s
a0 = 10
b0 = 0.5
data = []
for w in x:
    data = data + [-b0*w]
y_init = a0*np.exp(data)

#数据个数
datanum=len(y)
# 参数维数
paramdim=2
# 迭代最大次数
iterations=50
# LM算法的阻尼系数初值
lamda=0.01
# step1: 变量赋值
updateJ=1
a_tmp=a0
b_tmp=b0

#step2: 迭代
for i in range(iterations):
    if updateJ==1:
        # 根据当前估计值，计算雅克比矩阵
        J=np.zeros(datanum*paramdim).reshape(datanum,paramdim)
        for j in range(len(x)):
            J[j,:]=[np.exp(-b_tmp*x[j]),-a_tmp*x[j]*np.exp(-b_tmp*x[j])]
        # 根据当前参数，得到函数值
        tmp = []
        for w in x:
            tmp = tmp + [-b_tmp*w]
        y_tmp = a_tmp*np.exp(tmp)
        # 计算误差
        difference = y-y_tmp;
        # 计算（拟）海塞矩阵
        H = np.dot(J.T,J)
        # 若是第一次迭代，计算误差
        if i == 0:
            e=np.dot(difference,difference)

    # 根据阻尼系数lamda混合得到H矩阵
    H_lm = H+(lamda*np.eye(paramdim,paramdim));
    # 计算步长dp，并根据步长计算新的可能的\参数估计值
    dp = np.dot(np.mat(H_lm).I,np.dot(J.T,difference[:]))
    g = np.dot(J.T,difference[:])
    a_lm=a_tmp+dp[0,0]
    b_lm=b_tmp+dp[0,1]
    data2=[]
    for  w in x:
        data2 = data2 + [-b_lm*w]
    y_tmp_lm=a_lm*np.exp(data2)
    # 计算新的可能估计值对应的y和计算残差e
    difference_lm = y-y_tmp_lm
    e_lm=np.dot(difference_lm,difference_lm)
    
    if a_tmp == a_lm and b_tmp == b_lm:
        break
   
    # 根据误差，决定如何更新参数和阻尼系数
    if e_lm < e:
        lamda = lamda/10
        a_tmp = a_lm
        b_tmp = b_lm
        e = e_lm
        updateJ=1
    else:
        updateJ=0
        lamda=lamda*10
    print '第',i,'次迭代'
    print 'a=',a_tmp,'b=',b_tmp,'lamda=',lamda,'e=',e