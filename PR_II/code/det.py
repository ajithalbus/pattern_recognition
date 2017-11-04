import matplotlib.pyplot as plt
import matplotlib
import itertools
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def DETCurve(fps,fns,clr,label,pt=0):
    #online source
    
    axis_min = min(fps[0],fns[-1])
    ax_t.plot(fps,fns,clr,label=label)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Miss Rate')
    
    
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    ax_t.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_t.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_t.set_xticks(ticks_to_use)
    ax_t.set_yticks(ticks_to_use)
    plt.axis([0.001,1,0.001,1])
    
#det -fa,miss
#R
fig_t,ax_t = plt.subplots()

case1_fa=np.load('./op_data/fps-rl.npy')
case1_miss=np.load('./op_data/fns-rl.npy')


case1_fa=case1_fa/2
case1_miss=case1_miss/2
ERR=np.argmin(abs(case1_fa-case1_miss))

case2_fa=np.load('./op_data/fps-rl-sameC.npy')
case2_miss=np.load('./op_data/fns-rl-sameC.npy')


case2_fa=case2_fa/2
case2_miss=case2_miss/2
ERR2=np.argmin(abs(case2_fa-case2_miss))



case3_fa=np.loadtxt('./op_data/NLS/r-case3-fa.txt')
case3_miss=np.loadtxt('./op_data/NLS/r-case3-miss.txt')

case3_fa=3*case3_fa/2
case3_miss=3*case3_miss/2


case4_fa=np.loadtxt('./op_data/NLS/r-case4-fa.txt')
case4_miss=np.loadtxt('./op_data/NLS/r-case4-miss.txt')

case4_fa=3*case4_fa/2
case4_miss=3*case4_miss/2


case5_fa=np.loadtxt('./op_data/NLS/r-case5-fa.txt')
case5_miss=np.loadtxt('./op_data/NLS/r-case5-miss.txt')


case5_fa=3*case5_fa/2
case5_miss=3*case5_miss/2


plt.title('Real Data - DET')
DETCurve(case2_fa,case2_miss,clr='y',label='Bayes C-Same')
ax_t.plot([case2_fa[ERR2]],[case2_miss[ERR2]],'ko')


DETCurve(case1_fa,case1_miss,clr='m',label='Bayes C-Diff')
ax_t.plot([case1_fa[ERR]],[case1_miss[ERR]],'ko')



DETCurve(case3_fa,case3_miss,clr='r',label=u"Naive Bayes C = \u03c3^2*I")
ax_t.plot([case3_fa[499]],[case3_miss[499]],'ko')


DETCurve(case4_fa,case4_miss,clr='b',label='Naive Bayes C-Same')
ax_t.plot([case4_fa[499]],[case4_miss[499]],'ko')


DETCurve(case5_fa,case5_miss,clr='g',label='Naive Bayes C-Diff',pt=1)
ax_t.plot([case5_fa[499]],[case5_miss[499]],'ko',label='EER')
plt.legend()
plt.show()


#NLS
fig_t,ax_t = plt.subplots()

case1_fa=np.load('./op_data/fps-nls.npy')
case1_miss=np.load('./op_data/fns-nls.npy')


case1_fa=case1_fa/2
case1_miss=case1_miss/2
ERR=np.argmin(abs(case1_fa-case1_miss))

case2_fa=np.load('./op_data/fps-nls-sameC.npy')
case2_miss=np.load('./op_data/fns-nls-sameC.npy')


case2_fa=case2_fa/2
case2_miss=case2_miss/2
ERR2=np.argmin(abs(case2_fa-case2_miss))


case3_fa=np.loadtxt('./op_data/NLS/nls-case3-fa.txt')
case3_miss=np.loadtxt('./op_data/NLS/nls-case3-miss.txt')

case3_fa=3*case3_fa/2
case3_miss=3*case3_miss/2


case4_fa=np.loadtxt('./op_data/NLS/nls-case4-fa.txt')
case4_miss=np.loadtxt('./op_data/NLS/nls-case4-miss.txt')

case4_fa=3*case4_fa/2
case4_miss=3*case4_miss/2


case5_fa=np.loadtxt('./op_data/NLS/nls-case5-fa.txt')
case5_miss=np.loadtxt('./op_data/NLS/nls-case5-miss.txt')

case5_fa=3*case5_fa/2
case5_miss=3*case5_miss/2


plt.title('NLS - DET')


DETCurve(case2_fa,case2_miss,clr='y',label='Bayes C-Same')
ax_t.plot([case2_fa[ERR2]],[case2_miss[ERR2]],'ko')


DETCurve(case1_fa,case1_miss,clr='m',label='Bayes C-Diff')
ax_t.plot([case1_fa[ERR]+0.001],[case1_miss[ERR]],'ko')
print case1_fa[ERR],case1_miss[ERR]


DETCurve(case3_fa,case3_miss,clr='r',label=u"Naive Bayes C = \u03c3^2*I")
ax_t.plot([case3_fa[499]],[case3_miss[499]],'ko')


DETCurve(case4_fa,case4_miss,clr='b',label='Naive Bayes C-Same')
ax_t.plot([case4_fa[499]],[case4_miss[499]],'ko')


DETCurve(case5_fa,case5_miss,clr='g',label='Naive Bayes C-Diff',pt=1)
ax_t.plot([case5_fa[499]],[case5_miss[499]],'ko',label='EER')
plt.legend()
plt.show()

