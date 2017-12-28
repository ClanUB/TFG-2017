'''
Created on 10 nov. 2017

@author: Orquidea
'''
from sklearn.base import BaseEstimator
from utilFunc import KNN
from utilFunc import MAE
import pandas as pd
import numpy as np

class testRecom(BaseEstimator):
    def __init__(self, label = 0, k = 10, alpha = 2):
        self.label = label
        self.k = k
        self.alpha = alpha
        
    def __str__(self):
        return self.label + ": " + ", k: " + str(self.k) + ", alpha: " + str(self.alpha)
        
    def fit(self, X, y):
        self._X = X
        self._y = y
        
        return self
    
    def predict(self, X):
        predictions = []
        
        sim = []
        ar = []
        
        if type(X) == type(pd.DataFrame()):
            for student in X.index:
                pred,similar,area = self._recomender(X.loc[student])
                predictions.append(pred)
                sim.append(similar)
                ar.append(area)
            return np.round(pd.DataFrame(predictions,index = X.index), decimals = 1),sim,ar
        else:
            pred = self._recomender(X)
            return pred
    
    def _recomender(self, row):
        subjects = self._y.columns
        
        pairs = []
        similar_list = []
        area_list = []
        
        nn = KNN(self._X, row, self.k)
        
        prediction = pd.Series([0]*len(subjects),index = subjects)
        
        for subject in subjects:
            
            total_alpha = 0
            
            for similar_student,similarity in nn:
                if not np.isnan(self._y.iloc[similar_student][subject]) and self._y.iloc[similar_student][subject] != 0 and similarity > 0:
                    s = self._y.iloc[similar_student]
                    prediction.loc[subject] += (similarity**self.alpha)*(s[subject] - s.mean())/s.std()
                    similar_list.append((row.name,subject,s[subject]))
                    if self.alpha >= 0:
                        area_list.append((self.alpha,'g'))
                    else:
                        area_list.append((self.alpha,'r'))
                    total_alpha += abs(similarity**self.alpha)
                    
            prediction.loc[subject] *= row.std()
            prediction.loc[subject] /= total_alpha
            prediction.loc[subject] += row.mean()
            
        return prediction,similar_list,area_list
    
    def score(self,X,y):
        pred,_,_ = self.predict(X)
        return MAE(pred,y)
    