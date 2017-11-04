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

def polyfit(x,y,deg):
    X=[]
    for xi in x:
        line=[pow(xi,d) for d in range(deg+1)]
        X.append(line)
    X=np.array(X)
    Xt=np.transpose(X)
    Temp0=np.matmul(Xt,np.array(y)) #Xt.Y
    Temp1=np.matmul(Xt,X)           #Xt.X
    Temp1=np.linalg.inv(Temp1)      #(Xt.X)^-1
    Temp1=np.matmul(Temp1,Temp0)    #(Xt.X)^-1.Xt.Y
    #print Temp1
    return np.flipud(Temp1)         #fliping the coef


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
    ax1.set_title("Error vs Degree of polynomial")
    ax1.set_xlabel('X-axis ')
    ax1.set_ylabel('Y-axis ')

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
    #plot(k,error,'*r',label="Measured ")
    plot(k,error,'k',label="X-axis : degree\nY-axis : error")
    legend()
    #show()
    savefig("./result/reg/1d/normal0.png")

    #testing 
    print 'testing...'
    y_test_result=np.array([np.polyval(best_coef,k) for k in x_test])
    print 'error='+str(sq_error(y_test_result,y_test))

    print "Coef=>",best_coef;

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Training data and fitting curve")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    ax1.plot(x_train,y_train,'.',label="training data")
    ax1.plot(np.sort(x_train),np.polyval(best_coef,np.sort(x_train)),'-r',label="fitting curve") # plotting training data & reg curve
    legend()
    #show()
    savefig("./result/reg/1d/normal1.png")

    fig=figure()
    
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    
    ax1.set_title("Test output and Target output")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    ax1.plot(np.sort(x_test),np.polyval(best_coef,np.sort(x_test)),'-',label="Test Output")
    ax1.plot(x_test,y_test,'*r',label="Target Output")
    legend()
    #show()
    savefig("./result/reg/1d/normal2.png")
    
    print 'x-value\ttarget-y\toutcome-y'
    j=0
    for i in x_test:
        print i,'\t',y_test[j],'\t',np.polyval(best_coef,i)
        j=j+1

    return best_deg

def operations_ridge(x,y,l): #l=best deg
    #spliting data 70% train , 20% validate , 10% test
    x_train=x[:int(math.ceil(N*7/10))]
    x_valid=x[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x_test=x[int(math.ceil(N*9/10)):]

    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

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

    print 'training & validating...'
    for k in [-1, -0.8, -0.6, -0.4, -0.2, -0.1, -0.01, -0.001,0.001,0.01,0.1,0.2,0.4,0.6,0.8,1]:
    
        coef=polyfit_ridge(x_train,y_train,l,k)

        y_valid_result=np.array([np.polyval(coef,i) for i in x_valid]) #to calculate error

        error=sq_error(y_valid,y_valid_result)

        #print "k=",k," coef=>",coef
        
        avg=np.mean([int("{:E}".format(i).split('E')[1]) for i in coef])
        if math.fabs(avg)<min_error:
            min_error=math.fabs(avg)
            best_l=k
            #print best_l
            best_coef=coef
        #print k,error'''
        
        #ax1.plot(k,error,'.r')
        kerror.append([k,error])
        
        
        
        #print 'avg size(in powers of 10)=',avg 
        kavg.append([k,avg])
        #ax2.plot(k,avg,'.b')
    
    
    
    #print 'least error at lambda=',best_l,', error='+str(min_error)
    #fig.show()
    kerror=np.array(kerror)
    kavg=np.array(kavg)
    ax1.plot(kerror[:,0],kerror[:,1],'-r')
    ax1.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Error") # to make label alone
    ax1.legend()
    ax2.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Avg power of Coef")
    ax2.plot(kavg[:,0],kavg[:,1],'-b')
    ax2.legend()
    #show()
    fig.savefig("./result/reg/1d/ridge0.png")
    fig2.savefig("./result/reg/1d/ridge00.png")

    print 'Lowest power(abs) at lambda = ',best_l
    print 'Best Coef => ',best_coef
    
    #testing 
    print 'testing...'
    y_test_result=np.array([np.polyval(best_coef,k) for k in x_test])
    print 'error='+str(sq_error(y_test_result,y_test))

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Training data and fitting curve")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    ax1.plot(x_train,y_train,'.',label="Training data")
    ax1.plot(np.sort(x_train),np.polyval(best_coef,np.sort(x_train)),'-r',label="fitting curve") # plotting training data & reg curve
    legend()
    #show()
    savefig("./result/reg/1d/ridge1.png")

    fig=figure()
    
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    
    ax1.set_title("Test output and Target output")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    ax1.plot(np.sort(x_test),np.polyval(best_coef,np.sort(x_test)),'-',label="Test Output")
    ax1.plot(x_test,y_test,'*r',label="Target Output")
    legend();
    #show()
    savefig("./result/reg/1d/ridge2.png")
    
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
deg=operations(x,y,0)



#Ridge
print '\n\nridge regression...'
operations_ridge(x,y,deg)#deg)