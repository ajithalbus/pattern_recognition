import matplotlib.pyplot as plt
import matplotlib
import itertools
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import auc

    
def roc(c_fa,c_tp,clr,label):
    lw = 2
    plt.plot(c_fa, c_tp, color=clr,
            lw=lw, label=label+' (auc = %0.2f)' % float(1-auc(c_tp,c_fa)))#0)#roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.legend(loc="lower right")

plt.figure()
case1_tp=np.loadtxt('./op_data/R/fpr-real-sameC.csv')
case1_fa=np.loadtxt('./op_data/R/tpr-real-sameC.csv')
case2_tp=np.loadtxt('./op_data/R/fpr-real-diffC.csv')
case2_fa=np.loadtxt('./op_data/R/tpr-real-diffC.csv')

case3_fa=np.loadtxt('./op_data/NLS/r-case3-fa.txt')
case3_tp=np.loadtxt('./op_data/NLS/r-case3-tp.txt')
case4_fa=np.loadtxt('./op_data/NLS/r-case4-fa.txt')
case4_tp=np.loadtxt('./op_data/NLS/r-case4-tp.txt')
case5_fa=np.loadtxt('./op_data/NLS/r-case5-fa.txt')
case5_tp=np.loadtxt('./op_data/NLS/r-case5-tp.txt')

plt.title('Real Data - ROC')

roc(case1_fa,case1_tp,'y',label='Bayes-Same C')
roc(case2_fa,case2_tp,'k',label='Bayes-Different C')

roc(case3_fa,case3_tp,'r',label=u"Naive Bayes-C = \u03c3^2*I")
roc(case4_fa,case4_tp,'b',label='Naive Bayes-Same C')
roc(case5_fa,case5_tp,'g',label='Naive Bayes-Different C')

plt.show()

#NLS
plt.figure()
plt.title('NLS - ROC')

case1_tp=np.loadtxt('./op_data/NLS/fpr-nls-sameC.csv')
case1_fa=np.loadtxt('./op_data/NLS/tpr-nls-sameC.csv')
case2_tp=np.loadtxt('./op_data/NLS/fpr-nls-diffC.csv')
case2_fa=np.loadtxt('./op_data/NLS/tpr-nls-diffC.csv')
case3_fa=np.loadtxt('./op_data/NLS/nls-case3-fa.txt')
case3_tp=np.loadtxt('./op_data/NLS/nls-case3-tp.txt')
case4_fa=np.loadtxt('./op_data/NLS/nls-case4-fa.txt')
case4_tp=np.loadtxt('./op_data/NLS/nls-case4-tp.txt')


case5_fa=np.loadtxt('./op_data/NLS/nls-case5-fa.txt')
case5_tp=np.loadtxt('./op_data/NLS/nls-case5-tp.txt')

roc(case1_fa,case1_tp,'y',label='Bayes-Same C')
roc(case2_fa,case2_tp,'k',label='Bayes-Different C')
roc(case3_fa,case3_tp,'r',label=u"Naive Bayes-C = \u03c3^2*I")
roc(case4_fa,case4_tp,'b',label='Naive Bayes-Same C')

roc(case5_fa,case5_tp,'g',label='Naive Bayes-Different C')
plt.show()