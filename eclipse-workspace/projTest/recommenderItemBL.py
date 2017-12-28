from sklearn.base import BaseEstimator
from utilFunc import KNN
from utilFunc import MAE
import pandas as pd
import numpy as np

class RecommenderItemBL(BaseEstimator):
    def __init__(self, label = 0, k = 10, alpha = 2):
        self.label = label
        self.k = k
        self.alpha = alpha
        
    def __str__(self):
        return self.label + ": " + "k: " + str(self.k) + "alpha: " + str(self.alpha)
    
    def compute_bi(self,X,mu):
        
        bi = {}
        for subject in X.index:
            a = X.loc[subject]
            n = len(a)
            nans = a.isnull().sum()
            a = a.dropna()
            a = a - mu
            a = a.sum()
            bi[subject] = a/float(abs(n - nans))
        return bi
    
    def compute_bu(self,X,mu):
        
        bu = {}
        
        for student in X.columns:
            s = X[student]
            n = len(s)
            s = s - mu - list(self._bi_first.values())
            nans = s.isnull().sum()
            s = s.dropna()
            s = s.sum()
            bu[student] = s/float(abs(n - nans))
            
        return bu
        
    def fit(self, X, y):
        
        content_nn = {}
        
        self._X = X.T 
        self._y = y.T
        
        #Precompute the similarity of each second year module to each first year module
        for subject in self._y.index:
            content_nn[subject] = KNN(self._X, self._y.loc[subject], self.k)
            
        self._nn = content_nn
        
        self._mu_first = self._X.stack().mean()
        self._mu_second = self._y.stack().mean()
        
        self._bi_first = self.compute_bi(self._X,self._mu_first)
        self._bi_second = self.compute_bi(self._y,self._mu_second)
        
        return self
        

    def predict(self, X):
        
        X = X.T
        
        sim = []
        areas = []
        
        predictions = []
        
        if type(X) == type(pd.DataFrame()):
            
            self._bu_first = self.compute_bu(X,self._mu_first)
            
            for student in X.columns:
                #X[student] are the first year grades of student
                pred,similar,area = self.recommend(X[student],student)
                predictions.append(pred)
                sim.append(similar)
                areas.append(area)
                
            return np.round(pd.DataFrame(predictions,index = X.columns), decimals = 1),sim,areas
        
        else:
            pred = self.recommend(X)
            return pred
        
    def recommend(self,student,student_label):
        
        similar_list = []
        area_list = []
        
        #Get name of the modules to predict
        subjects = self._y.index
        prediction = pd.Series([0]*len(subjects),index = subjects)
        
        for subject in subjects:
            
            total_similarity = 0
            nn = self._nn[subject]
            bui = self._mu_second + self._bu_first[student_label] + self._bi_second[subject]
                
            for similar,similarity in nn:
                
                s = student.iloc[similar] #grade of the student for module "similar"

                #If the student has a grade for that module
                if not np.isnan(s):
                    
                    buj = self._mu_first + self._bu_first[student_label] + self._bi_first[self._X.index[similar]]
                    prediction.loc[subject] += (similarity**self.alpha)*(s - buj)
                    similar_list.append((student.name,subject,s))
                    total_similarity += abs(similarity**self.alpha)

            prediction.loc[subject] /= (total_similarity + 0.5)
            prediction.loc[subject] += bui
        
            if prediction.loc[subject] > 10: prediction.loc[subject] = 10
                
        return prediction,similar_list,area_list
    
    def score(self,X,y):
        pred,_,_ = self.predict(X)
        return -MAE(pred,y)