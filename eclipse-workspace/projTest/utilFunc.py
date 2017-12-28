import math as mt
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import pearsonr,spearmanr, kendalltau
from operator import itemgetter
from test.test_generators import syntax_tests
from dask import dataframe

#===============================================================================
# @Params:
# recom: instancia de un recomendador
# df1: df_year1
# df2: df_year2
# kf: instancia de un KFold
#
# @Return
# yPair_list: lista de parejas de predicted y e true y
# similar_list:
# area_list: 
#===============================================================================
#===============================================================================
# def crossValidation(recom, df1, df2, kf):
#     yPair_list = []
#     similar_list = []
#     area_list = []
#     
#     print (recom.label)
#     
#     for i, index in enumerate(kf):
#         
#         print ("Fold ",i)
#         
#         train_index = index[0]#list of index of training set, 45-220, eg.
#         test_index = index[1]#list of index of testing set, 0-44, eg.
# 
#         recom.fit(df1.iloc[train_index],df2.iloc[train_index])
#         
#         X_test = df1.iloc[test_index[:1]]
#         pred,similar,area = recom.predict(X_test)
# 
#         y_test = df2.iloc[test_index]
#         pred = pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)
#         
#         yPair_list.append((pred, y_test))
#         similar_list.append(similar)
#         area_list.append(area)
#     
#     return yPair_list,similar_list,area_list
#===============================================================================
def crossValidation(recommender,df1,df2,kf):
    
    predictions = []
    similar_list = []
    area_list = []
    
    print (recommender.label)
    
    for i,index in enumerate(kf):
        
        print ("Fold ",i)
        
        train = index[0]
        test = index[1]

        recommender.fit(df1.iloc[train],df2.iloc[train])
        
        f5 = df1.iloc[test[3]]
        pred,similar,area = recommender.predict(f5)

        pred = pd.DataFrame(pred,index = df2.iloc[test].index, columns = df2.iloc[test].columns)
        gt = df2.iloc[test]
        
        predictions.append((pred,gt))
        similar_list.append(similar)
        area_list.append(area)
    
    return predictions,similar_list,area_list

def getSimilarity(a,b):
     
    new_df = pd.concat([a,b],axis = 1).dropna(how = "any").T
    return new_df.iloc[0], new_df.iloc[1]
 
def KNN(df,row,k):
     
    neighbours = {}
    n = len(df)
     
    for i in range(n):
         
        a,b = getSimilarity(row,df.iloc[i])
         
        neighbours[i] = pearsonr(a,b)[0]
        if not np.isnan(neighbours[i]):
            neighbours[i] *= (min(len(a),n)/float(n))
         
    return sorted(neighbours.items(), key=itemgetter(1), reverse = True)[:k]
 
 
#===============================================================================
# df: X_train/self._X 
# row: X_test[i]
# k: k nearest neighbors
# 
# Retorna k nearest neighbors, cada elemento es una pareja de indice y puntuacion
# def KNN(df,row,k):
#     neighbours = {}
#     n = len(df)
#      
#     for i in range(n):
#         neighbours[i] = getSimilarity(row, df.iloc[i], n)
#          
#     return sorted(neighbours.items(), key=itemgetter(1), reverse = True)[:k]
#  
# #===============================================================================
# # a: una fila del dataframe X_test
# # b: una fila del dataframe X_train
# # n: len(X_train)
# #
# # p: pearson score
# #===============================================================================
# def getSimilarity(a, b, n):
#     new_df = pd.concat([a,b], axis=1).dropna(how = "any").T
#     p = pearsonr(new_df.iloc[0], new_df.iloc[1])[0]
#     if not np.isnan(p):
#         p *= min(len(a),n)/float(n)
#     return p
#===============================================================================


def MAE(df1,df2):
    df1 = df1.stack(dropna = False)
    df2 = df2.stack(dropna = False)
    a = abs(df1 - df2)
    nans = a.isnull().sum()
    return a.sum()/float(len(a) - nans)


def StandardDeviation(pred,Y):
    p = pred.stack(dropna = False)
    y = Y.stack(dropna = False)
    
    t = abs((p - y).dropna(how = 'any'))
    
    mean = t.mean()
    
    return mt.sqrt(((t - mean)**2).sum()/float(len(t)))


def getMarks(qual):
    marks = []
    for i in range(len(qual.index)):
        pc = qual.iloc[i]['nota_primera_conv']
        sc = qual.iloc[i]['nota_segona_conv']
        if pc >= sc: marks.append(pc)
        else: marks.append(sc)
    return pd.Series(marks,index = qual.index)


def crossValidationSM(estimator,df1,df2,kf):
    
    predictions = []
    similar_list = []
    area_list = []
    
    im = Imputer(missing_values='NaN', strategy='mean', axis=1)
    
    for i,index in enumerate(kf):
        
        train = index[0]
        test = index[1]
        
        pred = []
        ps = []
        
        for j in df2.iloc[train].columns:
            
            
            X = im.fit_transform(df1.iloc[train])
            y = im.fit_transform(df2.iloc[train][j])[0]
            X_test = im.fit_transform(df1.iloc[test])
            
            estimator.fit(X,y)

            prediction = pd.Series(np.round(estimator.predict(X_test),1),index = df2.iloc[test].index, name = j)
            pred.append(prediction)
       
        pred = pd.concat(pred, axis=1)
        gt = df2.iloc[test]
        
        predictions.append((pred,gt))
    
    return predictions