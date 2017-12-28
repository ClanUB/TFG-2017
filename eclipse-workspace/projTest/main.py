import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
from visualization import plotScatter
from loadData import loadData
from evaluation import evalRecommender
from evaluation2 import evalRecommender2
from recommenderUserM import RecommenderUserM

warnings.filterwarnings('ignore')
sys.path.append('../')

def main():
    pd.set_option('display.max_columns', None)    
    plt.rcParams['figure.figsize'] = (20, 20)
    ensenyament = "G1042"
    path = 'recommenderItemBL/' + ensenyament
     
    datas = loadData(ensenyament)
    primer = datas[0]
    segon = datas[1]
    lbl2 = datas[2]
    
    primer.fillna(value=5.0, inplace=True)
    segon.fillna(value=5.0, inplace=True)
    
    preds_eval1 = evalRecommender(primer, segon)
    #===========================================================================
    # evalRecommender2(primer, segon)
    #===========================================================================
    plotScatter(preds_eval1,"whitegrid",ensenyament,path,lbl2,primer,segon)

main()