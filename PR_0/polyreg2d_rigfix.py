#python 2.7.x
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import *
import math
from mpl_toolkits.mplot3d import Axes3D

def sq_error(A,B): # returns RMS error
    #print A,B
    j=0;
    sum=0;
    for i in A:
        sum=sum+(i-B[j])*(i-B[j])
        j=j+1
    #print sum,j
    return sum/j

def polyfit_deg2(x0,x1,y,deg,k=0):
    x=np.column_stack((x0,x1))
    X=[]
    for x0i,x1i in x:
        line =[[pow(x0i,d)*pow(x1i,e) for d in range(deg+1)] for e in range(deg+1)]
        line=np.array(line).flatten() # 2d matrix to 1d array
        np.reshape(line,line.shape[0])
        X.append(line)
    X=np.array(X)
    Xt=np.transpose(X) 
    Temp1=np.matmul(Xt,X) #Xt.X
    
    Temp1=Temp1+k*np.identity(Temp1.shape[0]) #Xt.X+kI
    Temp1=np.linalg.inv(Temp1) # (Xt.X+kI)^-1
    Temp0=np.matmul(Xt,np.array(y)) # Xt.Y
    Temp0=np.matmul(Temp1,Temp0) #(Xt.X+kI)^-1.Xt.Y
    return np.flipud(Temp0)

    


def polyvalxy(coef,x0,x1,deg):
    val=0
    for p in range(deg,-1,-1):
        for k in range(deg,-1,-1):
             val=val+pow(x0,k)*pow(x1,p)*coef[((deg+1)**2-1)-(p*3+k)]
    return val
   
    
    

def operations(x0,x1,y,l): #l=lamda
    #spliting data 70% train , 20% validate , 10% test
    x0_train=x0[:int(math.ceil(N*7/10))]
    x0_valid=x0[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x0_test=x0[int(math.ceil(N*9/10)):]

    x1_train=x1[:int(math.ceil(N*7/10))]
    x1_valid=x1[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x1_test=x1[int(math.ceil(N*9/10)):]


    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

    #reel start

    min_error=float('Inf')
    best_deg=3
    best_coef=[]

    fig=figure()
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    ax1.set_title("Error vs Degree of polynomial")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')

    print 'training & validating...'
    print 'deg','error'
    for dig in range(1,4):
    
        coef=polyfit_deg2(x0_train,x1_train,y_train,deg=dig)

        y_valid_result=np.array([polyvalxy(coef,k,p,deg=dig) for k,p in np.column_stack((x0_valid,x1_valid))]) #to calculate error

        error=sq_error(y_valid_result,y_valid)
        if error<min_error:
            min_error=error
            best_deg=dig
            best_coef=coef
        #print k,error
        plot(dig,error,'*r')
        
        print dig,error
    print 'least error at '+str(best_deg)+'deg , error='+str(min_error)
    plot(dig,error,'k',label="X-axis : degree\nY-axis : error")
    legend()
    savefig("./result/reg/2d/normal0.png")
    #show()

    #reel end
    #print polyfit_deg2(x0_train,x1_train,y_train)
    #-print 'training'
    #-best_coef=polyfit_deg2(x0_train,x1_train,y_train,deg=2)
    
    #validating 
    #-print 'validating...'
    #-y_valid_result=np.array([polyvalxy(best_coef,k,p,deg=2) for k,p in np.column_stack((x0_valid,x1_valid))])
    #-print 'error='+str(sq_error(y_valid_result,y_valid))
    
    
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0_train,x1_train,y_train,'.',label="Training data points")
    y_train_result=np.array([polyvalxy(best_coef,k,p,best_deg) for k,p in np.column_stack((x0_train,x1_train))])
    '''X, Y = np.meshgrid(x0_train, x1_train)
    zs = np.array([polyvalxy(best_coef,x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    '''
    ax.plot(x0_train,x1_train,y_train_result,'.r',label="fitting hyperplane")
    legend()
    show()

    #testing 
    print 'testing...'
    y_test_result=np.array([polyvalxy(best_coef,k,p,best_deg) for k,p in np.column_stack((x0_test,x1_test))])
    
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0_test,x1_test,y_test,'.',label="Target data points")
    ax.plot(x0_test,x1_test,y_test_result,'.r',label="Output data points")
    legend()
    show();
    '''
    fig=figure()
    
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    
    ax1.set_title("Model output(blue) and Target output(red)")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    ax1.plot(np.sort(x_test),np.polyval(best_coef,np.sort(x_test)),'-',x_test,y_test,'*r')
    show()
    '''
    print 'x0-val  x1-val  tar-y   out-y'
    j=0
    for i in x0_test:
        print "{0:0.4f}".format(i),'\t',"{0:0.4f}".format(x1_test[j]),'\t',"{0:0.4f}".format(y_test[j]),'\t',"{0:0.4f}".format(polyvalxy(best_coef,i,x1_test[j],l))
        j=j+1
    
    print 'error='+str(sq_error(y_test_result,y_test))
    return best_deg
                        

def operations_ridge(x0,x1,y,l): #l=deg
    #spliting data 70% train , 20% validate , 10% test
    x0_train=x0[:int(math.ceil(N*7/10))]
    x0_valid=x0[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x0_test=x0[int(math.ceil(N*9/10)):]

    x1_train=x1[:int(math.ceil(N*7/10))]
    x1_valid=x1[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    x1_test=x1[int(math.ceil(N*9/10)):]


    y_train=y[:int(math.ceil(N*7/10))]
    y_valid=y[int(math.ceil(N*7/10)):int(math.ceil(N*9/10))]
    y_test=y[int(math.ceil(N*9/10)):]

    #reel start

    min_error=float('Inf')
    best_l=0
    best_coef=[]

    fig=figure()
    ax1=fig.add_subplot(1,1,1)#((0.1,0.2,0.8,0.7))
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
    #print 'deg','error'
    for k in [-10,-1, -0.8, -0.6, -0.4, -0.2, -0.1, -0.01, -0.001,0.001,0.01,0.1,0.2,0.4,0.6,0.8,1,10]:
    
        coef=polyfit_deg2(x0_train,x1_train,y_train,deg=l,k=k)

        y_valid_result=np.array([polyvalxy(coef,u,p,deg=l) for u,p in np.column_stack((x0_valid,x1_valid))]) #to calculate error

        error=sq_error(y_valid_result,y_valid)
        
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
    ax1.plot(kerror[:,0],kerror[:,1],'-r')
    ax1.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Error") # to make label alone
    ax1.legend()
    ax2.plot(0,0,'k',label="X-axis : Lambda\nY-axis : Avg power of Coef")
    ax2.plot(kavg[:,0],kavg[:,1],'-b')
    ax2.legend()
    #show()
    fig.savefig("./result/reg/2d/ridge0.png")
    fig2.savefig("./result/reg/2d/ridge00.png")

    print 'Lowest power(abs) at lambda = ',best_l
    print 'Best Coef => ',best_coef


    #sys.exit()   
        
    
    #reel end
    #print polyfit_deg2(x0_train,x1_train,y_train)
    #-print 'training'
    #-best_coef=polyfit_deg2(x0_train,x1_train,y_train,deg=2)
    
    #validating 
    #-print 'validating...'
    #-y_valid_result=np.array([polyvalxy(best_coef,k,p,deg=2) for k,p in np.column_stack((x0_valid,x1_valid))])
    #-print 'error='+str(sq_error(y_valid_result,y_valid))
    
    
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0_train,x1_train,y_train,'.',label="Training data points")
    y_train_result=np.array([polyvalxy(best_coef,k,p,l) for k,p in np.column_stack((x0_train,x1_train))])
    '''X, Y = np.meshgrid(x0_train, x1_train)
    zs = np.array([polyvalxy(best_coef,x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    '''
    ax.plot(x0_train,x1_train,y_train_result,'.r',label="fitting hyperplane")
    legend()
    show()
    
    #testing 
    print 'testing...'
    y_test_result=np.array([polyvalxy(best_coef,u,p,deg=l) for u,p in np.column_stack((x0_test,x1_test))])
    
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0_test,x1_test,y_test,'.',label='Target data points')
    ax.plot(x0_test,x1_test,y_test_result,'.r',label='output data points')
    legend()
    show();
    '''
    fig=figure()
    
    ax1=fig.add_axes((0.1,0.2,0.8,0.7))
    
    ax1.set_title("Model output(blue) and Target output(red)")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    ax1.plot(np.sort(x_test),np.polyval(best_coef,np.sort(x_test)),'-',x_test,y_test,'*r')
    show()
    '''
    print 'x0-val  x1-val  tar-y   out-y'
    j=0
    for i in x0_test:
        print "{0:0.4f}".format(i),'\t',"{0:0.4f}".format(x1_test[j]),'\t',"{0:0.4f}".format(y_test[j]),'\t',"{0:0.4f}".format(polyvalxy(best_coef,i,x1_test[j],l))
        j=j+1
    
    print 'error='+str(sq_error(y_test_result,y_test))
                        
        

#main
with open('./dataset/q9_2.txt') as f :
    array=[[float(x) for x in line.split()] for line in f]  # getting values form file
array=np.array(array) # converting into np array
N=array.shape[0] # no of elements
x0,x1,y=np.split(array,3,1) # putting them in x0,x1 and y respectivly




x0=np.reshape(x0,N) # reshaping into single dimension
x1=np.reshape(x1,N) # reshaping into single dimension
y=np.reshape(y,N)  # reshaping into single dimension

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x0,x1,y)
show()
'''
#normal
print 'normal regression...'
deg=operations(x0,x1,y,0)



#Ridge

print 'ridge regression...l=0.2'
operations_ridge(x0,x1,y,deg)

