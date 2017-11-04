import numpy as np 
from matplotlib.pyplot import *

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

with open('./dataset/q9_1.txt') as f :
    array=[[float(x) for x in line.split()] for line in f]  # getting values form file
array=np.array(array) # converting into np array
N=array.shape[0] # no of elements
x,y=np.split(array,2,1) # putting them in x and y respectivly
x=np.reshape(x,N) # reshaping into single dimension
y=np.reshape(y,N)  # "
#z=np.polyfit(x,y,3) # curve fitting using builtin
z=polyfit(x,y,10)
#print z0
#print z1
plot(x,y,'.',x,np.polyval(z,x),'.r') # plotting curve
show()