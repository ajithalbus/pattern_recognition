#python 2.7.x
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import *
import math

def sq_error(A,B): # returns MS error
    #print A,B
    j=0;
    sum=0;
    for i in A:
        sum=sum+(i-B[j])*(i-B[j])
        j=j+1
    #print sum,j
    return sum/j



def polyfit_ridge(x,y,deg,k):  # k is lamda
    X=[]
    for xi in x:
        line=[pow(xi,d) for d in range(deg+1)]
        X.append(line)
    X=np.array(X)
    Xt=np.transpose(X) 
    Temp1=np.matmul(Xt,X) #Xt.X
    
    Temp1=Temp1+k*np.identity(Temp1.shape[0]) #Xt.X+kI
    Temp1=np.linalg.inv(Temp1) # (Xt.X+kI)^-1
    Temp0=np.matmul(Xt,np.array(y)) # Xt.Y
    Temp0=np.matmul(Temp1,Temp0) #(Xt.X+kI)^-1.Xt.Y
    return np.flipud(Temp0)

def operations(x,y,l): #l=lamda
    #spliting data 70% train , 20% validate , 10% test
    x_train=x[:int(math.ceil(N*7/10))]
    x_valid=x[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x_test=x[int(math.ceil(N*9/10)):]

    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

    min_error=float('Inf')
    best_deg=3
    best_coef=[]

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Error vs dim of polynomial")
    ax1.set_xlabel('X-axis(dim of poly)')
    ax1.set_ylabel('Y-axis(error)')

    print 'training & validating...'
    for k in range(1,15):
    
        coef=polyfit_ridge(x_train,y_train,k,l)

        y_valid_result=np.array([np.polyval(coef,i) for i in x_valid]) #to calculate error

        error=sq_error(y_valid,y_valid_result)
        if error<min_error:
            min_error=error
            best_deg=k
            best_coef=coef
        #print k,error
        plot(k,error,'*r')
    print 'least error at '+str(best_deg)+'deg , error='+str(min_error)
    show()

    #testing 
    print 'testing...'
    y_test_result=np.array([np.polyval(best_coef,k) for k in x_test])
    print 'error='+str(sq_error(y_test_result,y_test))

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Training data(blue) and fitting curve(red)")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    ax1.plot(x_train,y_train,'.', np.sort(x_train),np.polyval(best_coef,np.sort(x_train)),'-r') # plotting training data & reg curve
    show()

    fig=figure()
    
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    
    ax1.set_title("Model output(blue) and Target output(red)")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    ax1.plot(np.sort(x_test),np.polyval(best_coef,np.sort(x_test)),'-',x_test,y_test,'*r')
    show()
    
    print 'x-value\ttarget-y\toutcome-y'
    j=0
    for i in x_test:
        print i,'\t',y_test[j],'\t',np.polyval(best_coef,i)
        j=j+1

                        
        

#main
with open('./dataset/q9_1.txt') as f :
    array=[[float(x) for x in line.split()] for line in f]  # getting values form file
array=np.array(array) # converting into np array
N=array.shape[0] # no of elements
x,y=np.split(array,2,1) # putting them in x and y respectivly
x=np.reshape(x,N) # reshaping into single dimension
y=np.reshape(y,N)  # "
#normal
print 'normal regression...'
operations(x,y,0)



#Ridge
print 'ridge regression...'
operations(x,y,0.2)