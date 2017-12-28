from sklearn.cross_validation import KFold
from recommenderUserM import RecommenderUserM
from testRecom import testRecom
from utilFunc import crossValidation, MAE, StandardDeviation
import time

def evalRecommender(primer, segon):
    kf = KFold(len(primer.index), n_folds=5, random_state = 28392)
    #===========================================================================
    # recom = RecommenderItemBL(label = "BL")
    # title = "BL"
    #===========================================================================
    recom = testRecom(label="UserM", k=5)
    title = "UserM"
    
    start_time = time.time()
    preds_eval1, sim, area = crossValidation(recom,primer,segon,kf)
    t_eval1 = time.time() - start_time
        
        
    cv_mae = 0
    cv_std = 0
    
    for pred,gt in preds_eval1:
        cv_mae += MAE(pred,gt)
        cv_std += StandardDeviation(pred,gt)
        
    print (title,":\tcv_mae: %.3f"%(cv_mae/5), "\tcv_std: %.3f"%(cv_std/5), "\ttime: %.3f"%(t_eval1))    
    
    return preds_eval1