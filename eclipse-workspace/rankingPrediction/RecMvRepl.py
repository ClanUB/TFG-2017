import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from operator import itemgetter

class RecMvRepl(): 
    def __init__(self, name="Missing value replacement RecUserBased", kn=5):
        self.name = name
        self.kn = kn
        
        
    def fit(self, training_set):
        self.training_set = training_set
            
    
    def KNN(self, student):
        neighbours = {}
        n = len(self.training_set)
        
        for i in range(n):
            stuCompare = self.training_set.iloc[i]
            
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
        list_knn = sorted(neighbours.items(), key=itemgetter(1), reverse=True)[:kn]
        return list_knn
    
    
    def recommend(self, student):
        subjects = student.index
        list_knn = self.KNN(student)
        
        for subject in subjects:
            score = student[subject]
            if (np.isnan(score)):
                sum_p = 0
                sum_weighing = 0
                
                for i, p in list_knn:
                    sim_stu = self.training_set.iloc[i]
                    sum_p += p
                    sum_weighing += p*sim_stu[subject]
                    
                pred_score = sum_weighing / sum_p
                student[subject] = pred_score
                
                
    def predict(self, testing_set):
        multi_samples = type(testing_set) == type(pd.DataFrame())
            
        if multi_samples:        
            for i in range(len(testing_set)):
                student = testing_set.iloc[i]
                self.recommend(student)
        
        else:
            student = testing_set
            self.recommend(student)  