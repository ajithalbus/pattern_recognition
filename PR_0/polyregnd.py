import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import *
import math
from mpl_toolkits.mplot3d import Axes3D

def sq_error(A,B): # returns MS error
    #print A,B
    j=0;
    sum=0;
    for i in A:
        sum=sum+(i-B[j])*(i-B[j])
        j=j+1
    #print sum,j
    return sum/j

def polyfit(x,y,l=0,deg=1):
    X=[]
    for xi in x:
        t=np.array([1])
        t=np.append(t,pow(xi,deg))
        X.append(t)
    
    X=np.array(X)
    
    Xt=np.transpose(X)
    Temp1=np.matmul(Xt,X) #Xt.X
    
    #print np.matmul(Temp1,np.linalg.inv(Temp1))

    Temp1=Temp1+l*np.identity(Temp1.shape[0]) #Xt.X+kI
    Temp1=np.linalg.pinv(Temp1) # (Xt.X+kI)^-1 using psudo inv
    
    Temp0=np.matmul(Xt,np.array(y)) # Xt.Y
    Temp0=np.matmul(Temp1,Temp0) #(Xt.X+kI)^-1.Xt.Y
    #np.linalg.solve(X,np.array(y))
    return Temp0
    

def polyval(coef,x,deg):
    y=coef[0]
    
    for i in range(x.shape[0]):
        y=y+coef[i+1]*pow(x[i],deg)
    return y

            

def operations(x,y,l=0):
    x_train=x[:int(math.ceil(N*7/10))]
    x_valid=x[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x_test=x[int(math.ceil(N*9/10)):]

    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

    #training
    print 'training & validation...'
    min_error=float('Inf')
    best_coef=[];
    best_deg=1

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Error vs Degree of polynomial")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    print 'deg','error'
    for k in range(1,5):
        coef=polyfit(x_train,y_train,0,deg=k)
        y_valid_result=np.array([polyval(coef,i,k) for i in x_valid])
        #print y_valid[0],y_valid_result[0]
        error=sq_error(y_valid,y_valid_result)

        if error<min_error:
            min_error=error
            best_deg=k
            best_coef=coef
        print k,error
        #print k,error
        plot(k,error,'*r')
        

    print 'least error at '+str(best_deg)+' deg , error='+str(min_error)
    plot(k,error,'k',label="X-axis : degree\nY-axis : error")
    legend()
    savefig("./result/reg/nd/normal0.png")
    
    print "coef => ",best_coef
    print 'testing...'
    y_test_result=np.array([polyval(best_coef,i,best_deg) for i in x_test])

    error=sq_error(y_valid,y_valid_result)

    print 'Testing error=',error
    return best_deg

def operations_ridge(x,y,deg):
    x_train=x[:int(math.ceil(N*7/10))]
    x_valid=x[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x_test=x[int(math.ceil(N*9/10)):]

    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

    #training
    print 'training & validation...'

    min_error=float('Inf')
    best_l=0
    best_coef=[]

    fig=figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("Error vs Lambda")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    fig2=figure()
    ax2=fig2.add_subplot(1, 1, 1)
    ax2.set_title("Avg power of Coef vs Lambda")
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')

    kerror=[]

    kavg=[]

    for k in [0,1,10,100,1000,10000]:
        coef=polyfit(x_train,y_train,l=k,deg=deg)
    
    
        y_valid_result=np.array([polyval(coef,i,deg=deg) for i in x_valid])
        
        error=sq_error(y_valid,y_valid_result)

        avg=np.mean([int("{:E}".format(i).split('E')[1]) for i in coef])

        if math.fabs(avg)<min_error:
            min_error=math.fabs(avg)
            best_l=k
            #print best_l
            best_coef=coef
        #print k,error

        kerror.append([k,error])
        
        
        
        #print 'avg size(in powers of 10)=',avg 
        kavg.append([k,avg])
    
    kerror=np.array(kerror)
    kavg=np.array(kavg)
    #print kerror
    #print kavg
    
    ax1.plot(kerror[:,0],kerror[:,1],'-r')
    ax1.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Error") # to make label alone
    ax1.legend()
    ax2.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Avg power of Coef")
    ax2.plot(kavg[:,0],kavg[:,1],'-b')
    ax2.legend()
    #show()
    fig.savefig("./result/reg/nd/ridge0.png")
    fig2.savefig("./result/reg/nd/ridge00.png")

    print 'Lowest power(abs) at lambda = ',best_l
    print 'Best Coef => ',best_coef


        

    print 'testing...'
    y_test_result=np.array([polyval(coef,i,deg=deg) for i in x_test])

    error=sq_error(y_valid,y_valid_result)

    print 'Testing error=',error


with open('./dataset/9_3.txt') as f :
    dim=f.readline().split()[0]
    array=[[float(x) for x in line.split()] for line in f]  # getting values form file
array=np.array(array) # converting into np array

N=array.shape[0] # no of elements
print array.shape[1]
x=array[:,:array.shape[1]-2] #10-2
y=array[:,array.shape[1]-2] # putting them in x0,x1 and y respectivly

print 'normal regression...'
deg=operations(x,y,0)

print '\n\nridge regression...'
operations_ridge(x,y,deg)