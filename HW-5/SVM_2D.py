# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:44:49 2016

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr= open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

'''
SMO算法每次循环需要选择两个alpha进行优化处理，i是第一个alpha的下标
m是所有alpha的数目，随机选择出一个不等于输入值i的下标值
'''
def selectJrand(i,m):
    j=i 
    while (j==i):#选择与i不相等的j
        j = int(np.random.uniform(0,m))
    return j

'''
当alpha值太大或太小时进行相应调整
'''
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj
    
'''
计算w并画出分割面
'''    
def calcWs(alphas,dataArr,classLabels,b):
    X=np.mat(dataArr)
    #labelMat=np.mat(classLabels).transpos()
    tmp=np.mat(classLabels)
    labelMat=np.transpose(tmp)
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range (m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    min_x=min(X[:,0])[0,0]
    max_x=max(X[:,0])[0,0]
    y_min_x=float(-b-w[0]*min_x)/w[1]
    y_max_x=float(-b-w[0]*max_x)/w[1]
    plt.axis([0, 10, 0, 10])
    plt.axis([-2, 12, -10, 8])
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g',label='')    

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        #alphaPairsChanged记录alpha是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            #判断某数据向量是否可以被优化,是否违反KKT条件
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择第二个alpha
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #确保alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    print "j not moving enough"
                    continue
                    #update i by the same amount as j，but the update is in the oppostie direction
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): 
            iter += 1
        else: 
            iter = 0
        print "iteration number: %d" % iter
    return b,alphas

if __name__ == '__main__':

   dataArr,labelArr = loadDataSet('test.txt')
   b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)    

   for i in range(100):
       if alphas[i] > 0.0:
           print alphas[i]
           if  labelArr[i] == 1.0:
               plt.scatter(dataArr[i][0],dataArr[i][1],s=150,marker='*',c='r')
               print dataArr[i][0],dataArr[i][1]
           if  labelArr[i] == -1.0:
               plt.scatter(dataArr[i][0],dataArr[i][1],s=150,marker='*',c='b')
               print dataArr[i][0],dataArr[i][1]
       else:
           if labelArr[i] == 1:
               plt.scatter(dataArr[i][0],dataArr[i][1],s=50,marker='+',c='r')
           else:
               plt.scatter(dataArr[i][0],dataArr[i][1],s=50,marker='.',c='b')

   calcWs(alphas,dataArr,labelArr,b)   
   plt.legend()
   plt.show()