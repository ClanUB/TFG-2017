from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
import time
from utilFunc import crossValidationSM, MAE, StandardDeviation
from sklearn.cross_validation import KFold

def evalRecommender2(primer, segon):
    kf = KFold(len(primer.index), n_folds=5, random_state = 28392)

    estimators = [SVR(kernel = "rbf"),LinearRegression()]    
    titles = ["BL","SVR","LR"]
    predictions_eval2 = []
    times_eval2 = []
    
    for estimator in estimators:
        start_time = time.time()
        preds_eval2 = crossValidationSM(estimator,primer,segon,kf)
        predictions_eval2.append(preds_eval2)
        times_eval2.append(time.time() - start_time)
    
    for i, prediction in enumerate(predictions_eval2):
        
        cv_mae = 0
        cv_std = 0
        
        for pred,gt in prediction:
            
            cv_mae += MAE(pred,gt)
            cv_std += StandardDeviation(pred,gt)
            
    
    print (titles[i],":\tcv_mae: %.3f"%(cv_mae/5), "\tcv_std: %.3f"%(cv_std/5), "\ttime: %.3f"%(times_eval2[i]))    
