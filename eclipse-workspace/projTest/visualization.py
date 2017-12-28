import seaborn as sns
import pandas as pd
import numpy as np

len_kf = 5

def plotScatter(prediction,style,ensenyament,path,lbl2,primer,segon):
    sns.set_color_codes("deep")
        
    pred = []
    gt = []
    
    if ensenyament == "G1055": l = 4
    else: l = 0

    for fold in range(len_kf - l):
        pred.append(prediction[fold][0])
        gt.append(prediction[fold][1])

    pred = pd.concat(pred)
    gt = pd.concat(gt)

    means = []

    subjects = []

    for ind in pred.index:
        means.append(primer.loc[ind].mean())

    means = np.concatenate([[i]*len(segon.columns) for i in means])

    pred1 = np.array(pred.stack(dropna = False))
    gt1 = np.array(gt.stack(dropna = False))

    index = []

    for i in range(len(gt1)):
        if gt1[i] == 0.0: 
            index.append(i)

    l = lbl2*int(len(pred1)/len(segon.columns))

    pred = zip(means,pred1)
    gt = zip(means,gt1)

    pred2 = zip(l,pred1)
    gt2 = zip(l,gt1)

    pred1 = np.delete(pred1, index)
    gt1 = np.delete(gt1, index)

    #sns.set(style=style, color_codes=True)
    sns.set_style(style = style,rc = {"xtick.major.size": 10, "ytick.major.size": 10})

    g = (sns.jointplot(pred1, gt1, kind="reg",
                     xlim=(0, 10), ylim=(0, 10), color="b", size=10, stat_func = None, marginal_kws = dict(color = 'gray')).set_axis_labels("Prediction", "Ground Truth"))

    for i in range(0, 5):
        g.ax_joint.axvspan(i, i+1, ymin=0, ymax=.5, facecolor='r', alpha=0.4)
        
    for i in range(0,2):
        g.ax_joint.axhspan(5 + i, 5 + i +1, xmin = 0.5, xmax = 0.7, facecolor='b', alpha=0.4)
        
    for i in range(0,2):
        g.ax_joint.axhspan(7 + i, 7 + i +1, xmin = 0.7, xmax = 0.9, facecolor='g', alpha=0.4)
        
    g.ax_joint.axhspan(9, 10, xmin = 0.9, xmax = 1, facecolor='y', alpha=0.4)
    
    g.ax_marg_x.lines[0].set_color("black")
    g.ax_marg_y.lines[0].set_color("black")
    
    g.ax_joint.set_xticks(np.arange(0,11,1))
    g.ax_joint.set_yticks(np.arange(0,11,1))
    g.ax_joint.collections[0].set_visible(False)

    #Generate some colors and markers
    markers = ['o']*len(dict(pred))
    #colors = dict(zip(lbl2,np.random.random((10,3))))

    #Plot each individual point separatelY
    pred = list(dict(pred).items())
    for i in range(len(pred)):

        if pred[i][0] < 5:c = 'r'
        if pred[i][0] >= 5 and pred[i][0] < 7: c = 'b'
        if pred[i][0] >= 7 and pred[i][0] < 9: c = 'g'
        if pred[i][0] >= 9 and pred[i][0] <= 10: c = 'y'
        if gt[i][1] != 0:
                g.ax_joint.plot(pred[i][1], gt[i][1], color = c, marker = markers[i])
                
    
    g.ax_joint.plot([0, 10], [0, 10], 'k-', lw = 1)
    
    g.savefig(path + '/Scatter' + ensenyament, transparent=True, dpi = 350)