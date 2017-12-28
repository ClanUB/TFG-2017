from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from operator import itemgetter
from scipy.stats import pearsonr
import UtilFunctions

class RecUserBased(BaseEstimator):
    def __init__(self, name="User Based", kn=5, reverse=False, accMargin=2):
        self.name = name
        self.kn = kn
        self.reverse = reverse
        self.accMargin = accMargin
        
    
    def __str__(self):
        s = "name: "+self.name + "\tk neighbours: "+str(self.kn) + "\talpha: "+str(self.alpha) + "\taccMargin: "+str(self.accMargin)
        return s
        
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    
    def KNN(self, student):
        neighbours = {}
        n = len(self.X_train)
         
        for i in range(n):
            stuCompare = self.X_train.iloc[i]

            df_concat = pd.concat([student, stuCompare], axis=1)
            df_concat.dropna(how="any", inplace=True)
            
            if (len(df_concat)>=5):
                s1 = df_concat[df_concat.columns[0]]
                s2 = df_concat[df_concat.columns[1]]
                
                p = pearsonr(s1, s2)[0]
                if (not np.isnan(p)):
                    neighbours[i] = round(float(p), 4)
        
        if (len(neighbours)<self.kn):
            kn = len(neighbours)
        else:
            kn = self.kn
        list_knn = sorted(neighbours.items(), key=itemgetter(1), reverse = True)[:kn]
        return list_knn    
    
    
    def recommend(self, student):
        y_subjects = self.y_train.columns #class labels
        list_knn = self.KNN(student)
        
        prediction = pd.Series([0.0]*len(y_subjects), index=y_subjects) #Inicializacion del resultado
        
        for subject in y_subjects:
            sum_p = 0
            sum_weighing = 0
            for i, p in list_knn:
                sim_stu_y = self.y_train.iloc[i]
                if (not np.isnan(sim_stu_y[subject])):
                    sum_p += p
                    sum_weighing += p*sim_stu_y[subject]
            
            if (sum_p==0):
                print ("sum_p=0 at the student: ", student.name)
                if (self.reverse):
                    rank_score = float("-inf")
                else:
                    rank_score = float("inf")
            
            else: #Caso normal
                rank_score = sum_weighing / sum_p
                
                
            prediction[subject] = rank_score
        
        return prediction
    
    
    #===========================================================================
    # Predecir el ranking de las asignaturas para varios alumnos o un solo alumno
    # X_test: DataFrame(varios alumnos) or Series(un solo alumno) 
    # 
    # df_pred: resultado de predicciones tipo DataFrame or
    # pred: resultado de prediccion unitaria tipo Series
    #===========================================================================
    def predict(self, X_test):
        if type(X_test) == type(pd.DataFrame()):
            predictions = []
            
            for studentId in X_test.index:
                stud_row = X_test.loc[studentId]
                pred = self.recommend(stud_row) #Predecir via recomendacion
                
                predictions.append(pred)
            
            df_pred = pd.DataFrame(predictions, index=X_test.index)
            df_pred_r = UtilFunctions.score2ranking(df_pred, reverse=self.reverse)
            return df_pred_r 
        
        else:
            pred = self.recommend(X_test)
            df_pred = pd.DataFrame(pred, index=X_test.index)
            df_pred_r = UtilFunctions.score2ranking(df_pred, reverse=self.reverse)
            return df_pred_r
        

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        df_error = abs(y_pred-y_test)
        error = df_error.sum().sum() / float(df_error.shape[0] * df_error.shape[1])
        return error
        