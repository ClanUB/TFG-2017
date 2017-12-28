'''
Created on 8 nov. 2017

@author: Orquidea
'''
from sklearn.base import BaseEstimator
from utilFunc import KNN
from utilFunc import MAE
import pandas as pd
import numpy as np

class RecommenderUserM(BaseEstimator):
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
    
    #===========================================================================
    # Predecir la nota de las asignaturas para todos los alumnos nuevos
    # X: X_test 
    # 
    # df_pred: dataframe de predicciones
    # similarities: 
    # areas:
    #===========================================================================
    def predict(self, X):
        predictions = []
        similities = []
        areas = []
        
        if type(X) == type(pd.DataFrame()):
            for studentID in X.index:
                stud_row = X.loc[studentID]
                pred, similar, area = self.recommend(stud_row)
                 
                predictions.append(pred)
                similities.append(similar)
                areas.append(area)
                
            df_pred = np.round(pd.DataFrame(predictions, index=X.index), decimals=1)
            return df_pred, similities, areas
        
        else:
            pred = self.recommend(X)
            return pred
    
    
    #===========================================================================
    # Predecir la nota de las asignaturas que va a sacar un alumno nuevo en funcion de los k vecinos mas cercanos.
    # row: X_test[i]
    #===========================================================================
    def recommend(self, row):
        subjects = self._y.columns
        
        similar_list = []
        area_list = []
        
        nn = KNN(self._X, row, self.k)
        
        prediction = pd.Series([0]*len(subjects), index = subjects)
        
        for subject in subjects:
            
            total_alpha = 0
            
            for sim_stu_ind, sim_score in nn:
                sim_stu = self._y.iloc[sim_stu_ind]
                if not np.isnan(sim_stu[subject]) and sim_stu[subject] != 0 and sim_score > 0:
                    score = (sim_score**self.alpha)*(sim_stu[subject] - sim_stu.mean())/sim_stu.std()
                    prediction.loc[subject] += score
                    similar_list.append((row.name,subject,sim_stu[subject]))
                    
                    if self.alpha >= 0:
                        area_list.append((self.alpha,'g'))
                    else:
                        area_list.append((self.alpha,'r'))
                    total_alpha += abs(sim_score**self.alpha)
                    
            prediction.loc[subject] *= row.std()
            prediction.loc[subject] /= total_alpha
            prediction.loc[subject] += row.mean()
            
        return prediction, similar_list, area_list
    
    def score(self,X,y):
        pred,_,_ = self.predict(X)
        return MAE(pred,y)
    