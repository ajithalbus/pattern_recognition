import numpy as np
import scipy.io
import scipy.misc
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import random

def b2i(x):
    if x=='' or x=='-':
        return 0
    return int(x,2)

def splitter(x,order=0):
    x=str(b(int(np.real(x))))
    x=x[::-1] # reverse
    C=x[:8]
    B=x[8:16]
    A=x[16:]
    C=C[::-1]
    B=B[::-1]
    A=A[::-1]
        
    if order==0:
        return b2i(A),b2i(B),b2i(C)
    if order==1:
        return b2i(B),b2i(C),b2i(A)
    if order==2:
        return b2i(C),b2i(A),b2i(B)
        
def rect2sq(img):
    return np.matmul(np.transpose(img),img)

def b(x):
    
    return "{0:b}".format(x)

def rgb2cat(img,order=0):
    #print (img[0,0,0])
    tmp=np.empty((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if order==0:
                tmp[i,j]=int(str(b(img[i,j,0]))+str(b(img[i,j,1]))+str(b(img[i,j,2])),2)
            if order==1:
                tmp[i,j]=int(str(b(img[i,j,1]))+str(b(img[i,j,2]))+str(b(img[i,j,0])),2)
            if order==2:
                tmp[i,j]=int(str(b(img[i,j,2]))+str(b(img[i,j,0]))+str(b(img[i,j,1])),2)
                
                
    return rect2sq(tmp)#np.reshape(tmp,(img.shape[0],img.shape[1],1))


def rgb2green(img):
    
    green=np.copy(img)
    green=green[:,:,1]
    return rect2sq(green)

def rgb2red(img):
    
    red=np.copy(img)
    red=red[:,:,0]
    return rect2sq(red)


def rgb2blue(img):
    
    blue=np.copy(img)
    blue=blue[:,:,2]
    return rect2sq(blue)

            

def rgb2gray(rgb):
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #wiki

    return rect2sq(gray)

def read_image(path):
    image = scipy.misc.imread(path)
    return image
'''
def write_image(image,colour,path='./evdsq.jpg',order=0):
    
    new_image=np.zeros((image.shape[0],image.shape[0],3))
    #print new_image
    if colour=='red':
        new_image[:,:,0]=image
    if colour=='green':
        new_image[:,:,1]=image
    if colour=='blue':
        new_image[:,:,2]=image
    if colour=='grey':
        new_image=image
    if colour=='cat':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_image[i,j]=splitter(image[i,j])

    new_image = np.clip(np.real(new_image), 0, 255).astype('uint8')
    #print new_image
    scipy.misc.imsave(path, new_image)
'''

def evd(img,colour,order=-1,per=""):
    val,vec=np.linalg.eig(img)
    '''
    idx = val.argsort()   
    val = val[idx]
    vec = vec[:,idx]
    '''
    t_val=np.copy(val)
    t_val=np.absolute(t_val)
    idx = t_val.argsort()[::-1]  
    val = val[idx]
    vec = vec[:,idx]

    result=np.zeros((0,2))
    
    
    #random N
    print 'RANDOM - N'
    n=1

    #while n<=val.shape[0]:
    for n in [10,25,50,100,125,150,175,200,225,250,300,350,400,450,500,525,530,535,540,545,548]:
        tmp=np.copy(val)
        #print 'N='+str(n)
        for i in random.sample(xrange(0,tmp.shape[0]),tmp.shape[0]-n):
            tmp[i]=0
        #print val
        rec=np.matmul(np.matmul(vec,np.diag(tmp)),np.linalg.inv(vec))
        error=np.linalg.norm(img-np.real(rec))/np.linalg.norm(img)
        print str(n)+','+str(math.fabs(error))
        #print 'Error='+str(math.fabs(error))
        #print rec
        #write_image(rec,colour,path='./result/sq/evd_rand_'+colour+"_"+per+"_"+str(n)+'.jpg',order=order)
        
        result=np.append(result,[[n,error]],axis=0)
        #print result
        '''
        if n<(val.shape[0]/2):
            n=n*2
        else:
            n=n+32
        '''
    #print np.matrix(result)

    fig=plt.figure()
      
    ax1=fig.add_subplot(1, 1, 1)
        
    ax1.set_title("RAND-N(red),TOP-N(blue),BOT-N(green) vs error colour:"+colour+" order:"+per)
    ax1.set_xlabel('X-axis(N)')
    ax1.set_ylabel('Y-axis(Error)')
    ax1.plot(result[:,0],result[:,1],'-r')
    

    #top-N
    print 'TOP-N'
    result=np.zeros((0,2))
    n=1

    for n in [10,25,50,100,125,150,175,200,225,250,300,350,400,450,500,525,530,535,540,545,548]:
        tmp=np.copy(val)
        #print 'N='+str(n)
        for i in range(n,tmp.shape[0]):
            tmp[i]=0
        #print val
        rec=np.matmul(np.matmul(vec,np.diag(tmp)),np.linalg.inv(vec))
        error=np.linalg.norm(img-np.real(rec))/np.linalg.norm(img)
        print str(n)+','+str(math.fabs(error))
        #print 'Error='+str(math.fabs(error))
        #print rec
        #write_image(rec,colour,path='./result/sq/evd_top_'+colour+"_"+per+"_"+str(n)+'.jpg',order=order)
        
        result=np.append(result,[[n,error]],axis=0)
        #print result
        '''
        if n<(val.shape[0]/2):
            n=n*2
        else:
            n=n+32
        '''
    #print np.matrix(result)

    ax1.plot(result[:,0],result[:,1],'-')
    
    #top-N
    print 'BOTTOM-N'
    result=np.zeros((0,2))
    n=1

    for n in [10,25,50,100,125,150,175,200,225,250,300,350,400,450,500,525,530,535,540,545,548]:
        tmp=np.copy(val)
        #print 'N='+str(n)
        for i in range(0,tmp.shape[0]-n):
            tmp[i]=0
        #print val
        rec=np.matmul(np.matmul(vec,np.diag(tmp)),np.linalg.inv(vec))
        error=np.linalg.norm(img-np.real(rec))/np.linalg.norm(img)
        #print 'Error='+str(math.fabs(error))
        print str(n)+','+str(math.fabs(error))
        #print rec
        #write_image(rec,colour,path='./result/sq/evd_buttom_'+colour+"_"+per+"_"+str(n)+'.jpg',order=order)
        
        result=np.append(result,[[n,error]],axis=0)
        #print result
        '''
        if n<(val.shape[0]/2):
            n=n*2
        else:
            n=n+32
        '''
    #print np.matrix(result)

    ax1.plot(result[:,0],result[:,1],'-g')
    
    #savefig('./result/rect/graph'+colour+"_"+per+'.png')
    #show()        

img=read_image('./dataset/rect.jpg')



#greyscale
print '--GREYSTYLE--'
grey=rgb2gray(img)
print grey.shape
evd(grey,'grey')


#red
print '--RED--'
red=rgb2red(img)
evd(red,'red')

#green
print '--GREEN--'
red=rgb2green(img)
evd(red,'green')

#blue
print '--BLUE--'
red=rgb2blue(img)
evd(red,'blue')

#cat
print '--CAT-RGB--'
cat=rgb2cat(img,0) #rgb
evd(cat,'cat',order=0,per="RGB")

print '--CAT-GBR--'
cat=rgb2cat(img,1) #gbr
evd(cat,'cat',order=1,per="GBR")

print '--CAT-BRG--'
cat=rgb2cat(img,2) #brg
evd(cat,'cat',order=2,per="BRG")

