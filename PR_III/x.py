
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import seaborn as sns
from matplotlib import cm
import sys
import random
import os
import mpmath as mp
from multiprocessing import Process
from multiprocessing import Pool
from sklearn.cluster import KMeans
mp.mp.dps=64



# In[2]:


listing = os.listdir('./data_set/mountain')


class1=np.empty((1,23))
for infile in listing:
    tmp=np.loadtxt('./data_set/mountain/'+infile)
    tmp=np.average(tmp,axis=0)
    class1 = np.vstack((class1, tmp))
    
listing = os.listdir('./data_set/street')


class2=np.empty((1,23))
for infile in listing:
    tmp=np.loadtxt('./data_set/street/'+infile)
    tmp=np.average(tmp,axis=0)
    class2 = np.vstack((class2, tmp))
    
listing = os.listdir('./data_set/tallbuilding')


class3=np.empty((1,23))
for infile in listing:
    tmp=np.loadtxt('./data_set/tallbuilding/'+infile)
    tmp=np.average(tmp,axis=0)
    class3 = np.vstack((class3, tmp))


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


    
'''class1=np.loadtxt('./data_set/syn/class1.txt')
class2=np.loadtxt('./data_set/syn/class2.txt')
np.random.shuffle(class1)
np.random.shuffle(class2)
'''


# In[3]:


class Gaussian:
    
    def __init__(self, mu, sigma):
        
        self.mu = mu
        self.sigma = sigma
        #print mu,sigma
    def pdf_log(self,datum):
        size = len(datum)
        det = np.linalg.det(self.sigma)

        N =  ((2*math.pi)**size * det)**0.5 
        x_mu = (datum - self.mu)
        inv = np.linalg.inv(self.sigma)        
        result = -0.5 * np.matmul(x_mu , np.matmul(inv, np.transpose(x_mu)))
        return  result-np.log(N)
    
    def pdfx(self,datum):
        size = len(datum)
        
        datum=mp.matrix(datum)
        sigma=mp.matrix(self.sigma)
        mu=mp.matrix(self.mu)
        det = mp.det(sigma)
        
        N =  ((2*math.pi)**size * det)**0.5
        
        x_mu = (datum - mu)
        
    
        
        inv = sigma**-1        
        
        #print N,np.array(x_mu.T.tolist()).shape,np.array(inv.tolist()).shape
        result = -0.5 * x_mu.T * (inv* x_mu)
        #print result.tolist()[0][0]
        result=mp.power(math.e,result.tolist()[0][0])
        return  result/N
    
    def pdf(self, datum):
        size = len(datum)
        if(is_invertible(self.sigma)):
            det = np.linalg.det(self.sigma)
        else:
            det = self.sigma.diagonal().prod()
            
        N =  ((2*math.pi)**size * det)**0.5 
        x_mu = (datum - self.mu)
	if is_invertible(self.sigma):
        	inv = np.linalg.inv(self.sigma)
	else:
		inv = np.linalg.pinv(self.sigma)
        result = math.pow(math.e, -0.5 * np.matmul(x_mu , np.matmul(inv, np.transpose(x_mu))))
        return  result/N
    
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1})'.format(self.mu, self.sigma)
    


# In[4]:


N=class1.shape[0]

c1_train=class1[:int(math.ceil(N*7/10))]
c1_valid=class1[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c1_test=class1[int(math.ceil(N*9/10)):]


N=class2.shape[0]
c2_train=class2[:int(math.ceil(N*7/10))]
c2_valid=class2[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c2_test=class2[int(math.ceil(N*9/10)):]

N=class3.shape[0]
c3_train=class3[:int(math.ceil(N*7/10))]
c3_valid=class3[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
c3_test=class3[int(math.ceil(N*9/10)):]


# In[5]:


c1_mu=np.mean(c1_train,axis=0)
temp=c1_train.T
c1_sig=np.cov(temp)

c2_mu=np.mean(c2_train,axis=0)
temp=c2_train.T
c2_sig=np.cov(temp)

c3_mu=np.mean(c3_train,axis=0)
temp=c3_train.T
c3_sig=np.cov(temp)





# In[6]:


class GaussianMixture:
    
    def __init__(self, data,n):
        self.data = data
        self.N=n
        kmeans = KMeans(n_clusters=n,random_state=0).fit(data)
        
        mu=[]
        for i in range(n):
            mu.append(kmeans.cluster_centers_[i])
        
        #print mu
        '''
        c=[]
        for k in range(n):
            c.append(np.array([i for i in data if kmeans.predict([i])[0]==k]))
        '''
        
        
        
        
        sig=[]
        
        for i in range(n):
            temp=data.T
            sig.append(np.cov(temp))
            
        
        
        self.gas=[]
        for i in range(n):
            self.gas.append(Gaussian(mu[i],sig[i]))
        
        
        self.w = np.full(n,1.0/n)
        
        self.logN=0
    def E(self):
        v=[]
        self.logN=0
        gamma=[]
        for i in self.data:
            gamma=[]
            for k in range(self.N):
                gamma.append(self.gas[k].pdf(i)*self.w[k])
            
            
            total_gamma = sum(gamma)
            
            
            
            for k in range(self.N):
                gamma[k]/=total_gamma
            
            
            #for k in range(self.N):
             #   self.logN += self.gas[k].pdf(i)*self.w[k]
            
            v.append(gamma)
            
        return v

    def M(self,gammas):
        Nk=[]
        for i in range(self.N):
            Nk.append(sum(gammas[:,i]))
            
        
        
        for k in range(self.N):
            self.gas[k].mu=np.sum(i*j for (i,j) in zip(gammas[:,k], self.data))/Nk[k]
        
        for k in range(self.N):
            self.gas[k].sigma=np.sum([(i* (np.outer((j - self.gas[k].mu),np.array(j - self.gas[k].mu)))) for (i,j) in zip(gammas[:,k], self.data)],axis=0)/Nk[k]
            self.gas[k].sigma=np.clip(self.gas[k].sigma,a_min=1e-3,a_max=1e50)
        for k in range(self.N):
            
            self.w[k]=Nk[k]/len(self.data)
            
        

    def __str__(self):
        return str(self.mu,self.sig)
    
    def likelihood(self,x):
        val=0
        for i in range(self.N):
            val+=self.gas[i].pdf(x)*(self.w[i])
        return val
    
    def likelihood_test(self,x):
        val=0
        for i in range(self.N):
            val+=self.gas[i].pdf(x)*(self.w[i])
        return val
    
    


# In[7]:


def trainer(iters,mixnumber):
    global mix
    
    for i in range(iters):
        v=mix[mixnumber].E()
        v=np.array(v)
        mix[mixnumber].M(v)
    return

def testerl(mixnumber):
    if mixnumber==0:
        samples=c1_test
    elif mixnumber==1:
        samples=c2_test
    else:
        samples=c3_test
    total=0
    tx=0
    for s in samples:
        total+=1
        tmp=[mix[0].likelihood_test(s),mix[1].likelihood_test(s),mix[2].likelihood_test(s)]
        i=tmp.index(max(tmp))
        if i==mixnumber:
            tx+=1
    return tx*1.0/total


        


# In[8]:


accs=[]
mix=[]
clusters=1
iters = 2

oldN=0

mix.append( GaussianMixture(c1_train,clusters))#c2_train,4)

thread0 = Process(
            target=trainer,
            name="class0",
            args=[iters, 0],
            )
thread0.start()



oldN=0
mix.append(GaussianMixture(c2_train,clusters))#c2_train,4)

thread1 = Process(
            target=trainer,
            name="class1",
            args=[iters, 1],
            )
thread1.start()



oldN=0
mix.append( GaussianMixture(c3_train,clusters))#c2_train,4)
thread2 = Process(
            target=trainer,
            name="class2",
            args=[iters, 2],
            )
thread2.start()

thread0.join()
thread1.join()
thread2.join()

print 'trained...'




# In[9]:


for m in mix:
    for g in m.gas:
        g.mu=np.array(g.mu.tolist(),dtype=np.float64)
        g.sigma=np.array(g.sigma.tolist(),dtype=np.float64)


p=Pool()
a=p.map(testerl, [0,1,2])
print a
print 'c = ',clusters*2,' accuracy = ',a
accs.append([clusters*2,np.mean(a)])






ax = plt.gca()
accs=np.array(accs)

ax.plot(accs[:,0],accs[:,1])
plt.savefig('./results/real_full',format='eps')
    
plt.show()



# accs=[]
# mix=[]
# for clusters in [2]:
#     iters = 1
#     #clusters=1
#     train_data=np.vstack([c1_train,c2_train])
#     test_data=np.vstack([c1_valid,c2_valid])
#     
#     oldN=0
#     
#     mix.append( GaussianMixture(c1_train,clusters))#c2_train,4)
#     
#     thread0 = Process(
#                 target=trainer,
#                 name="class0",
#                 args=[iters, 0],
#                 )
#     thread0.start()
#     
#         
#     
#     oldN=0
#     mix.append(GaussianMixture(c2_train,clusters))#c2_train,4)
#     
#     thread1 = Process(
#                 target=trainer,
#                 name="class1",
#                 args=[iters, 1],
#                 )
#     thread1.start()
#     
#     
# 
#     oldN=0
#     mix.append( GaussianMixture(c3_train,clusters))#c2_train,4)
#     thread2 = Process(
#                 target=trainer,
#                 name="class2",
#                 args=[iters, 2],
#                 )
#     thread2.start()
#     
#     thread0.join()
#     thread1.join()
#     thread2.join()
#     
#     print 'trained...'
#     
#     
#     
#     with Pool(3) as p:
#         a=np.mean(p.map(tester, [(c1_test,0),(c2_test,1),(c3_test,2)]))
#     '''
#     total=0
#     crt=0
#     wrg=0
#     tx=0
#     
#     samples=c1_valid
#     for s in samples:
#         total+=1
#         tmp=[mix[0].likelihood(s),mix[1].likelihood(s),mix[2].likelihood(s)]
#         i=tmp.index(max(tmp))
#         if i==0:
#             tx+=1
#         #print 0,i
#     
#             
#         
#     crt=tx
#     wrg=total-crt
# 
#     total_crt=crt
#     total_wrg=wrg
# 
# 
#     total=0
#     crt=0
#     wrg=0
#     tx=0
#     samples=c2_valid
#     for s in samples:
#         total+=1
#         tmp=[mix[0].likelihood(s),mix[1].likelihood(s),mix[2].likelihood(s)]
#         i=tmp.index(max(tmp))
#         if i==1:
#             tx+=1
#         print 1,i
#             
#         
#     crt=tx
#     wrg=total-crt
#     
#     total_crt+=crt
#     total_wrg+=wrg
#     
#     total=0
#     crt=0
#     wrg=0
#     tx=0
#     samples=c3_valid#+c2_valid
#     for s in samples:
#         total+=1
#         tmp=[mix[0].likelihood(s),mix[1].likelihood(s),mix[2].likelihood(s)]
#         i=tmp.index(max(tmp))
#         if i==2:
#             tx+=1
#         print 2,i    
#         
#     crt=tx
#     wrg=total-crt
# 
#     total_crt+=crt
#     total_wrg+=wrg
#     '''
#     
#     print 'c = ',clusters*2,' accuracy = ',a
#     accs.append([clusters*2,a])
#     
# 
#     
# 
# 
#     if a==1.0:
#         break
# 
# 
# ax = plt.gca()
# accs=np.array(accs)
# 
# ax.plot(accs[:,0],accs[:,1])
# plt.savefig('./results/real_full',format='eps')
#     
# plt.show()
# 
# 

# In[10]:


'''while (True):
    a=str(raw_input())
    test=np.average(np.loadtxt('./data_set/'+a),axis=0)
    
    
    tmp=[mix1.likelihood(s),mix2.likelihood(s),mix3.likelihood(s)]
    i=tmp.index(max(tmp))
        
    if i==0:
        print 'mountain'
    elif i==1:
        print 'street'
    else:
        print 'tallBuilding'

'''

