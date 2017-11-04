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
    X = np.linspace(mu[0]-3*sig[0][0]**0.5, mu[0]+3*sig[0][0]**0.5, N)
    Y = np.linspace(mu[1]-3*sig[1][1]**0.5, mu[1]+3*sig[1][1]**0.5, N)

    X, Y = np.meshgrid(X, Y)

    zs = np.array([pdf([x,y],mu,sig) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z,alpha=1, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=color)

    cset = ax.contourf(X, Y, Z, alpha=1,zdir='z', offset=-0.1, cmap=color)

    ax.set_zlim(-0.1,0.07)

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
    



data=np.loadtxt('../data_set/9_ls.txt')
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

#iX,iY=get_intersection([M0,C0],[M1,C1])

#abline(M0,C0)
#abline(M1,C1)
#abline(M2,C2)




g1w1,g1w0=gi(c1_mu,sigSq)
g2w1,g2w0=gi(c2_mu,sigSq)
g3w1,g3w0=gi(c3_mu,sigSq)


waste,c1_eigVec=np.linalg.eig(sigSq*np.identity(2))
c1_eigVec=20*c1_eigVec
ax.plot(np.array([c1_mu[0],c1_eigVec[0][0]+c1_mu[0]]),np.array([c1_mu[1],c1_eigVec[0][1]+c1_mu[1]]),np.array([-0.1,-0.1]),'r')
ax.plot(np.array([c1_mu[0],c1_eigVec[1][0]+c1_mu[0]]),np.array([c1_mu[1],c1_eigVec[1][1]+c1_mu[1]]),np.array([-0.1,-0.1]),'r')


waste,c2_eigVec=np.linalg.eig(sigSq*np.identity(2))
c2_eigVec=20*c2_eigVec
ax.plot(np.array([c2_mu[0],c2_eigVec[0][0]+c2_mu[0]]),np.array([c2_mu[1],c2_eigVec[0][1]+c2_mu[1]]),np.array([-0.1,-0.1]),'b')
ax.plot(np.array([c2_mu[0],c2_eigVec[1][0]+c2_mu[0]]),np.array([c2_mu[1],c2_eigVec[1][1]+c2_mu[1]]),np.array([-0.1,-0.1]),'b')

waste,c3_eigVec=np.linalg.eig(sigSq*np.identity(2))
c3_eigVec=20*c3_eigVec
ax.plot(np.array([c3_mu[0],c3_eigVec[0][0]+c3_mu[0]]),np.array([c3_mu[1],c3_eigVec[0][1]+c3_mu[1]]),np.array([-0.1,-0.1]),'g')
ax.plot(np.array([c3_mu[0],c3_eigVec[1][0]+c3_mu[0]]),np.array([c3_mu[1],c3_eigVec[1][1]+c3_mu[1]]),np.array([-0.1,-0.1]),'g')



avg_mu=(c1_mu+c2_mu+c3_mu)/3
avg_sig=sigSq*np.identity(2)

N = 60
X = np.linspace(avg_mu[0]-6*avg_sig[0][0]**0.5, avg_mu[0]+6*avg_sig[0][0]**0.5, N)
Y = np.linspace(avg_mu[1]-6*avg_sig[1][1]**0.5, avg_mu[1]+6*avg_sig[1][1]**0.5, N)

X, Y = np.meshgrid(X, Y)

for i in zip(np.ravel(X), np.ravel(Y)):
    i=np.array(i)
    t=max(np.matmul(g1w1,i)+g1w0,np.matmul(g2w1,i)+g2w0,np.matmul(g3w1,i)+g3w0)
    if t==np.matmul(g1w1,i)+g1w0:
        plt.plot([i[0]],[i[1]],[-0.05],'.r')
    elif t==np.matmul(g3w1,i)+g3w0:
        plt.plot([i[0]],[i[1]],[-0.05],'.g')
    else:
        plt.plot([i[0]],[i[1]],[-0.05],'.b')

    


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

