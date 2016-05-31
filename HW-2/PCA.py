# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:23:15 2016

@author: Dell
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
  
    
"""
参数：
 - dataMatrix：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征 
 - k：表示取前k个特征值对应的特征向量
返回值：
 - finalData：参数一指的是返回的低维矩阵，对应于输入参数二
 - reconData：参数二对应的是移动坐标轴后的矩阵
"""
def pca(dataMatrix, k):
    #第一步分别求x和y的平均值，然后对于所有的样例，都减去对应的均值。
    #计算均值,行表示样本数，列表示特征 
    average = np.mean(dataMatrix,axis=0) 
    m, n = np.shape(dataMatrix)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = dataMatrix - avgs
    #计算协方差矩阵
    covX = np.cov(data_adjust.T) 
    print np.shape(covX)
    #求解协方差矩阵的特征值和特征向量
    featValue, featVec=  np.linalg.eig(covX)
     #按照featValue进行从大到小排序
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print "k must lower than feature number"
        return
    else:
        #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData  

def writeCharList(charlist):
    fp = open ('optdigits-orig_3.tra', 'a')
    for item in charlist:
        fp.write(item)
        #print item
    fp.write('\n')
    fp.close()

def getSamples():
    row_count = 33
    k = 0
    charList = []
    
    fileHandle = open ('optdigits-orig.tra')  
    fileList = fileHandle.readlines()
    if os.path.exists('optdigits-orig_3.tra') is True:
        os.remove('optdigits-orig_3.tra')
        
    for fileLine in fileList:
        row_count -= 1
        if row_count != 0:
            charList.append(fileLine)  # as a buffer of the matrix
            #print fileLine
        else:
            row_count = 33
            if fileLine == ' 3\n':
                k += 1
                writeCharList(charList) # write the three sample to the file
            charList = []
    fileHandle.close()
    print('have already got all the Threes from original files.')
    
def getTrainMatrix():
    feature, featureMatrix = [], []
    
    fileHandle = open ('optdigits-orig_3.tra')  
    fileList = fileHandle.readlines()
    for fileLine in fileList:
        line = fileLine.rstrip()
        #print line
        if line != "":
            feature += [int(x) for x in line]
        else:
            #print len(feature)
            featureMatrix.append(feature)
            feature = []
    
    #print len(featureMatrix)
    dataMatrix = np.array(featureMatrix)
    #print dot(dataMatrix,dataMatrix.T)  
    print('have already transformed the training samples into matrixs.')
    print(np.shape(dataMatrix))
    #fp = open ('result', 'a')
    #fp.write(str(featureMatrix))
    #fp.close()
    return dataMatrix
      
def main():    
    getSamples()
    t=getTrainMatrix()
    k = 2
    return pca(t, k)
if __name__ == "__main__":
    finalData, reconMat = main()
    print np.shape(finalData)
    dataArr1 = np.array(finalData)
    
    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
    plt.plot(axis_x1, axis_y1,'.')
    plt.xlabel('First Principal Component'); 
    plt.ylabel('Second Principal Component');
    plt.show()