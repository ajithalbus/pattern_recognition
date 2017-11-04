import numpy as np 
import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

iX,iY=0,0
fig = plt.figure()
plt.hold(True)
ax = fig.gca(projection='3d')
f=2 #trick to draw boundry

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    global f
    f=f-1
    #print axes.get_xlim()[0]
    x_vals = np.array([axes.get_xlim()[0],iX])
    if(f==0):
        f=2
        x_vals = np.array([iX,axes.get_xlim()[1]])
        
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')
    

def plotr(mu,sig,color):
    
    N = 60
    X = np.linspace(mu[0]-30, mu[0]+30, N)
    Y = np.linspace(mu[1]-30, mu[1]+30, N)

    X, Y = np.meshgrid(X, Y)

    zs = np.array([pdf([x,y],mu,sig) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z,alpha=0.5, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=color)

    cset = ax.contourf(X, Y, Z, alpha=0.5,zdir='z', offset=-0.1, cmap=color)

    ax.set_zlim(-0.1,0.001)

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
    



#data=np.loadtxt('../data_set/9_ls.txt')
c1=np.loadtxt('../data_set/9/class1.txt')
c2=np.loadtxt('../data_set/9/class2.txt')
c3=np.loadtxt('../data_set/9/class3.txt')
random.shuffle(c1)
random.shuffle(c2)
random.shuffle(c3)

N=1000
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

iX,iY=get_intersection([M0,C0],[M1,C1])

#abline(M0,C0)
#abline(M1,C1)
#abline(M2,C2)


plt.show()


g1w1,g1w0=gi(c1_mu,sigSq)
g2w1,g2w0=gi(c2_mu,sigSq)
g3w1,g3w0=gi(c3_mu,sigSq)

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

np.savetxt('../op_data/NLS/nls-case3-fa.txt',c_fa)
np.savetxt('../op_data/NLS/nls-case3-miss.txt',c_miss)
np.savetxt('../op_data/NLS/nls-case3-tp.txt',c_tp)
