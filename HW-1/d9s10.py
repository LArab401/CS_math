# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:50:12 2016

@author: Dell
"""

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#the curve of y=sin(x)
def func(x):
    return np.sin(2*np.pi*x)

def func_d9(x,p):
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9= p
    return w0+w1*x+w2*x**2+w3*x**3+w4*x**4+w5*x**5+w6*x**6+w7*x**7+w8*x**8+w9*x**9

# 实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
def residuals_d9(p, y, x):
    return y - func_d9(x, p)
    
x0 = np.linspace(0, 1)# 真实数据的函数参数
y0 = func(x0) # 真实数据

x1 = np.linspace(0, 1, 10)# 模拟噪声数据
y1 = func(x1)
y2 = y1 + 0.4*np.random.randn(len(x1)) # 加入噪声之后的实验数据

p0 = [1,1,1,1,1,1,1,1,1,1]
# 调用leastsq进行曲线拟合
# p0为拟合参数的初始值
# args为需要拟合的实验数据
plsq_d9 = leastsq(residuals_d9, p0, args=(y2, x1))

plot1,=plt.plot(x0,y0,'g')
plot2,=plt.plot(x1,y2,'b*')
plot3,=plt.plot(x0,func_d9(x0,plsq_d9[0]),'r')
plt.xlabel('x')
plt.ylabel('t')
plt.legend([plot1,plot2,plot3],('original','sample_dots','fitted curve'))
plt.show()