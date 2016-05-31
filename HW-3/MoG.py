# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:43:29 2016

@author: Dell
"""

import numpy as np
import pylab as plt
from numpy import *
from numpy.linalg import inv,det

def Gaussian_2d(mean,var,num):
      points = []
      points.append(mean[0] + np.random.randn(num) * var[0])
      points.append(mean[1] + np.random.randn(num) * var[1])
      return points
        
def posterior_probability(x,pmu,detpsigma,invpsigma): #计算N（xi | μk,Σk）    
    D = len(x)
    t = -0.5 * dot(dot((x-pmu).T,invpsigma),x-pmu) #-1/2*(x-μ)T*Σ^-1(x-μ)######
    posprob = (2*pi)**(-D/2) * (detpsigma**(-0.5)) * exp(t)
    return posprob
    
def EM_for_MoG(x):
    #变量初始化    
    classnum = 2 #两个类
    datanum,dim = x.shape #记录数据个数以及维度
    print datanum,dim
    pmu = zeros((classnum,dim)) #初始μ为0
    psigma = zeros((classnum,dim,dim)) #σ初始化
    detpsigma = zeros(classnum) #σ的行列式
    invpsigma = psigma.copy() #σ的逆
    pi=zeros(classnum) #被选为某个高斯分布的概率
    gamma=zeros((datanum,dim)) #第i个数据由滴第k个分布生成的概率
    for k in range(classnum):
        pmu[k]=random.rand(dim)
        psigma[k]=eye(dim,dim)
        invpsigma[k]=inv(psigma[k])
        detpsigma[k]=det(psigma[k])
        pi[k]=random.rand()
    pi=pi/sum(pi) #归一化
    iteration = 0
    while True:
        if iteration > 100: #迭代100次理论上是应该看似然函数是否收敛
            break
        #E-step计算第i个数据由滴第k个分布生成的概率
        for i in range(datanum):
            for k in range(classnum):
                postpro = posterior_probability(x[i],pmu[k],detpsigma[k],invpsigma[k])
                gamma[i,k] = pi[k]*postpro
            gamma[i] = gamma[i]/sum(gamma[i])
        #M-Step通过极大似然估计可以通过求到令参数=0得到参数pmu，psigma的值
        pmu = zeros((classnum,dim))
        psigma = zeros((classnum,dim,dim))
        NK = sum(gamma,0)
        pi = NK/datanum
        for k in range(classnum):
            for i in range(datanum):
                pmu[k] = pmu[k] + gamma[i][k]*x[i]
            pmu[k] = pmu[k]/NK[k]
            for i in range(datanum):
                temp = mat(x[i]-pmu[k])
                psigma[k] = psigma[k] + gamma[i][k]*dot(temp.T,temp)
            psigma[k] = psigma[k]/NK[k]
            detpsigma[k]=det(psigma[k])
            invpsigma[k]=inv(psigma[k])
        iteration = iteration+1
    for k in range(classnum):
        print 'CLASS:',k
        print 'mu'
        print pmu[k]
        print 'sigma'
        print psigma[k]
    
    plt.title('MOG after '+str(iteration)+' iteration(s)')
    idx=zeros(datanum)
    color=['m*','b+']
    for i in range(datanum):
        idx[i] = nonzero(gamma[i] == max(gamma[i]))[0][0]
    for k in range(classnum):
        t = nonzero(idx==k)[0]
        plt.plot(x[t,0],x[t,1],color[k],label="group"+str(k+1))
    plt.legend()
    plt.show()
    
mean1 = (2.5,2)
var1 = (0.7,0.7)
mean2 = (6,4)
var2 = (0.7,0.7)
data1 = Gaussian_2d(mean1,var1,100)
data2 = Gaussian_2d(mean2,var2,100)
x=vstack([hstack([data1[0],data2[0]]) , hstack([data1[1],data2[1]])])
EM_for_MoG(x.T)   