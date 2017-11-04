import numpy as np 
import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.metrics import auc

iX,iY=0,0
fig = plt.figure()
plt.hold(True)
ax = fig.gca(projection='3d')
f=2 #trick to draw boundry

def DETCurve(fps,fns,clr,pt=0):
    #online source
    fig_t,ax_t = plt.subplots()

    axis_min = min(fps[0],fns[-1])
    ax_t.plot(fps,fns,clr)
    
    plt.yscale('log')
    plt.xscale('log')
    
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    ax_t.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_t.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_t.set_xticks(ticks_to_use)
    ax_t.set_yticks(ticks_to_use)
    plt.axis([0.001,1,0.001,1])
    
    if pt==1:
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #credits - sklearn
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format((cm[i, j]), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    global f
    f=f-1
    #print axes.get_xlim()[0]
    x_vals = np.array(axes.get_xlim())
        
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')
    

def plotr(mu,sig,color):
    
    N = 60
    X = np.linspace(mu[0]-3*sig[0][0]**0.5, mu[0]+3*sig[0][0]**0.5, N)
    Y = np.linspace(mu[1]-3*sig[1][1]**0.5, mu[1]+3*sig[1][1]**0.5, N)

    X, Y = np.meshgrid(X, Y)

    zs = np.array([pdf([x,y],mu,sig) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z,alpha=0.5, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=color)

    cset = ax.contourf(X, Y, Z, alpha=0.5,zdir='z', offset=-0.00001, cmap=color)

    ax.set_zlim(-0.00001,0.000001)

    ax.set_zticks(np.linspace(0,0.1,5))
    ax.view_init(27, -21)

def get_intersection(line1, line2):
    
    l1=np.insert(line1, 1, -1)
    l2=np.insert(line2, 1, -1)
    x, y, z= np.cross(l1, l2)
    return np.hstack([x, y]) / z
    
def DB(mu1,mu2,sigSq):
    w=mu1-mu2
    x0=0.5*(mu1+mu2)
    return (w,x0)
    
def gi(mu,sigSq):
    w1=mu.T/sigSq
    w0=-np.matmul(mu.T,mu)/(2*sigSq)+np.log(1.0/3)
    return w1,w0

    

def pdf(x, mu, sigma):
    size = len(x)
    det = np.linalg.det(sigma)
    
    N =  ((2*math.pi)**size * det)**0.5 
    x_mu = (x - mu)
    inv = np.linalg.inv(sigma)        
    result = math.pow(math.e, -0.5 * np.matmul(x_mu , np.matmul(inv, np.transpose(x_mu))))
    return  result/N
    



data=np.loadtxt('../data_set/group_9.txt')
c1=data[0:500]
c2=data[500:1000]
c3=data[1000:1500]
random.shuffle(c1)
random.shuffle(c2)
random.shuffle(c3)

N=500
c1_train=c1[:int(math.ceil(N*7/10))]
c1_valid=c1[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c1_test=c1[int(math.ceil(N*9/10)):]

c2_train=c2[:int(math.ceil(N*7/10))]
c2_valid=c2[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c2_test=c2[int(math.ceil(N*9/10)):]

c3_train=c3[:int(math.ceil(N*7/10))]
c3_valid=c3[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c3_test=c3[int(math.ceil(N*9/10)):]


c1_mu=np.mean(c1_train,axis=0)
temp=np.vstack((c1_train[:,0],c1_train[:,1]))
c1_sig=np.cov(temp)

c2_mu=np.mean(c2_train,axis=0)
temp=np.vstack((c2_train[:,0],c2_train[:,1]))
c2_sig=np.cov(temp)

c3_mu=np.mean(c3_train,axis=0)
temp=np.vstack((c3_train[:,0],c3_train[:,1]))
c3_sig=np.cov(temp)

sigSq=np.mean(np.diagonal((c1_sig+c2_sig+c3_sig))/3)

plotr(c1_mu,sigSq*np.identity(2),cm.inferno)
plotr(c2_mu,sigSq*np.identity(2),cm.viridis)
plotr(c3_mu,sigSq*np.identity(2),cm.plasma)



#G1
#print sigSq,c1_mu,c2_mu


w,x0=DB(c1_mu,c2_mu,sigSq)
M0=-w[0]/w[1]
C0=(w[0]*x0[0]+w[1]*x0[1])/w[1]

w,x0=DB(c2_mu,c3_mu,sigSq)
M1=-w[0]/w[1]
C1=(w[0]*x0[0]+w[1]*x0[1])/w[1]

w,x0=DB(c3_mu,c1_mu,sigSq)
M2=-w[0]/w[1]
C2=(w[0]*x0[0]+w[1]*x0[1])/w[1]





g1w1,g1w0=gi(c1_mu,sigSq)
g2w1,g2w0=gi(c2_mu,sigSq)
g3w1,g3w0=gi(c3_mu,sigSq)


waste,c1_eigVec=np.linalg.eig(sigSq*np.identity(2))
c1_eigVec=1000*c1_eigVec
ax.plot(np.array([c1_mu[0],c1_eigVec[0][0]+c1_mu[0]]),np.array([c1_mu[1],c1_eigVec[0][1]+c1_mu[1]]),np.array([-0.00001,-0.00001]),'r')
ax.plot(np.array([c1_mu[0],c1_eigVec[1][0]+c1_mu[0]]),np.array([c1_mu[1],c1_eigVec[1][1]+c1_mu[1]]),np.array([-0.00001,-0.00001]),'r')



c2_eigVec=c1_eigVec
ax.plot(np.array([c2_mu[0],c2_eigVec[0][0]+c2_mu[0]]),np.array([c2_mu[1],c2_eigVec[0][1]+c2_mu[1]]),np.array([-0.00001,-0.00001]),'b')
ax.plot(np.array([c2_mu[0],c2_eigVec[1][0]+c2_mu[0]]),np.array([c2_mu[1],c2_eigVec[1][1]+c2_mu[1]]),np.array([-0.00001,-0.00001]),'b')

waste,c3_eigVec=np.linalg.eig(c3_sig)
c3_eigVec=c1_eigVec
ax.plot(np.array([c3_mu[0],c3_eigVec[0][0]+c3_mu[0]]),np.array([c3_mu[1],c3_eigVec[0][1]+c3_mu[1]]),np.array([-0.00001,-0.00001]),'g')
ax.plot(np.array([c3_mu[0],c3_eigVec[1][0]+c3_mu[0]]),np.array([c3_mu[1],c3_eigVec[1][1]+c3_mu[1]]),np.array([-0.00001,-0.00001]),'g')



avg_mu=(c1_mu+c2_mu+c3_mu)/3
avg_sig=sigSq*np.identity(2)

N = 60
X = np.linspace(avg_mu[0]-4*avg_sig[0][0]**0.5, avg_mu[0]+4*avg_sig[0][0]**0.5, N)
Y = np.linspace(avg_mu[1]-4*avg_sig[1][1]**0.5, avg_mu[1]+4*avg_sig[1][1]**0.5, N)

X, Y = np.meshgrid(X, Y)

for i in zip(np.ravel(X), np.ravel(Y)):
    i=np.array(i)
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g1w1,i)+g1w0:
        plt.plot([i[0]],[i[1]],[-0.000005],'.r')
    elif t==np.matmul(g3w1,i)+g3w0:
        plt.plot([i[0]],[i[1]],[-0.000005],'.b')
    else:
        plt.plot([i[0]],[i[1]],[-0.000005],'.g')

    


plt.show()


tp=0
fp=0
miss=0
count=0
fa=0
print 'testing class1'
for i in c1_test:
    count=count+1
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g1w1,i)+g1w0:
        tp=tp+1
    else:
        fp=fp+1

print 'testing class2'
for i in c2_test:
    count=count+1
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g2w1,i)+g2w0:
        tp=tp+1
    else:
        fp=fp+1
    
print 'testing class3'
for i in c3_test:
    count=count+1
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g3w1,i)+g3w0:
        tp=tp+1
    else:
        fp=fp+1

tpr=float(tp)/count
fpr=float(fp)/count

print 'tpr =',tpr
print 'fpr =',fpr 

#CONFUSION
cf=np.zeros((3,3))
#cf_per=np.empty((4,4))
count=0
#print 'testing class1'
print c1_test.shape
for i in c1_test:
    count=count+1
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g1w1,i)+g1w0:
        cf[0,0]+=1
        #plt.plot([i[0]],[i[1]],'.g')
    elif t==np.matmul(g2w1,i)+g2w0:
        cf[0,1]+=1
    else:
        cf[0,2]+=1
        #plt.plot([i[0]],[i[1]],'.r')

#print 'testing class2'
for i in c2_test:
    
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g2w1,i)+g2w0:
        cf[1,1]+=1
        #plt.plot([i[0]],[i[1]],'.g')
    elif t==np.matmul(g1w1,i)+g1w0:
        cf[1,0]+=1
    else:
        cf[1,2]+=1
        #plt.plot([i[0]],[i[1]],'.r')
    
#print 'testing class3'
for i in c3_test:
    
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g3w1,i)+g3w0:
        cf[2,2]+=1
        #plt.plot([i[0]],[i[1]],'.g')
    elif t==np.matmul(g1w1,i)+g1w0:
        cf[2,0]+=1
    else:
        cf[2,1]+=1
        #plt.plot([i[0]],[i[1]],'.r')

print cf
plot_confusion_matrix(cf,['class_1','class_2','class_3'],normalize=True)
plt.show()
#DET and ROC
#DETC3
tp=0
fp=0
c3_tp=[]

miss=0
fa=0
c3_misses=[]
c3_fa=[]
count=0
temp=g3w0
g3w0=g3w0-10
for i in range(1000):
    g3w0=g3w0+0.02
    for i in c3_valid:
        count=count+1
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t!=np.matmul(g3w1,i)+g3w0:
            miss=miss+1
        else:
            tp=tp+1
            
    
    for i in c1_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g3w1,i)+g3w0:
            
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')
        
    
    for i in c2_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g3w1,i)+g3w0:
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')

    #print fa,count    
    fa=float(fa)/(3*count)
    miss=float(miss)/(3*count)
    tp=float(tp)/count
    #print fa,miss
    c3_fa.append(fa)
    c3_misses.append(miss)
    c3_tp.append(tp)
    #print fa,miss
    fa=0
    miss=0
    tp=0
    count=0

g3w0=temp

#DETC1
tp=0
c1_tp=[]
miss=0
fa=0
c1_misses=[]
c1_fa=[]
count=0
temp=g1w0
g1w0=g1w0-10
for i in range(1000):
    g1w0=g1w0+0.02
    for i in c1_valid:
        count=count+1
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t!=np.matmul(g1w1,i)+g1w0:
            miss=miss+1
        else:
            tp=tp+1
            
    
    for i in c2_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g1w1,i)+g1w0:
            
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')
        
    
    for i in c3_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g1w1,i)+g1w0:
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')

    #print fa,count    
    fa=float(fa)/(3*count)
    miss=float(miss)/(3*count)
    tp=float(tp)/count

    #print fa,miss
    c1_fa.append(fa)
    c1_misses.append(miss)
    c1_tp.append(tp)
    #print fa,miss
    fa=0
    tp=0
    miss=0
    count=0
g1w0=temp

#DETC2
tp=0
c2_tp=[]
miss=0
fa=0
c2_misses=[]
c2_fa=[]
count=0
temp=g2w0
g2w0=g2w0-10
for i in range(1000):
    g2w0=g2w0+0.02
    for i in c2_valid:
        count=count+1
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t!=np.matmul(g2w1,i)+g2w0:
            miss=miss+1
        else:
            tp=tp+1
            
    
    for i in c1_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g2w1,i)+g2w0:
            
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')
        
    
    for i in c3_valid:
        count=count
        t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
        if t==np.matmul(g2w1,i)+g2w0:
            fa=fa+1
            #plt.plot([i[0]],[i[1]],'.g')

    #print fa,count    
    fa=float(fa)/(3*count)
    miss=float(miss)/(3*count)
    tp=float(tp)/count

    #print fa,miss
    c2_fa.append(fa)
    c2_tp.append(tp)
    c2_misses.append(miss)
    #print fa,miss
    fa=0
    miss=0
    count=0

g2w0=temp

#plt.plot(c1_fa,c1_misses)
#plt.show()
#DET plotter
c_fa=(np.array(c1_fa)+np.array(c2_fa)+np.array(c3_fa))/3
c_miss=(np.array(c1_misses)+np.array(c2_misses)+np.array(c3_misses))/3
c_tp=(np.array(c1_tp)+np.array(c2_tp)+np.array(c3_tp))/3

#c_fa=np.loadtxt('../fa.csv')
#c_miss=np.loadtxt('../miss.csv')
DETCurve(c_fa,c_miss,'r',pt=1);

np.savetxt('../op_data/NLS/r-case3-fa.txt',c_fa)
np.savetxt('../op_data/NLS/r-case3-miss.txt',c_miss)
np.savetxt('../op_data/NLS/r-case3-tp.txt',c_tp)


plt.figure()
lw = 2
plt.plot(c_fa, c_tp, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % float(1-auc(c_tp,c_fa)))#0)#roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




