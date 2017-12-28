#===============================================================================
# from pylab import *
# x, y = arange(10), [cos(2)]*10
# plot(x, y)
# show()
#===============================================================================
# #===============================================================================
# from matplotlib import pyplot
# 
# 
# from matplotlib import pyplot as plt
# import numpy as np
# 
# x, y = np.arange(10), [np.cos(2)]*10
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x,y)
# plt.show()
#===============================================================================
from sklearn import metrics
import pandas as pd

def compute_accuracy(df1, df2):   
    df_diff = abs(df1 - df2)
    n = (df_diff==0).sum().sum()
    accuracy = n / float(df1.shape[0] * df1.shape[1])
    return accuracy


def main():
    d1 = {"a": [0,1,2], "b": [1,2,3], "c": [2,3,4]}
    d2 = {"a": [1,2,3], "b": [1,2,3], "c": [4,5,6]}
    
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    
    compute_accuracy(df1, df2)

    print("hola", \
          "adios")
main()