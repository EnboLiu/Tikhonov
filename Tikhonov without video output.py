"""
==========================
Tikhonov rugularization of video data

program with 3D wireframe plot

Author: Enbo Liu

Date: 12/20/2017

The data is porcessed frame by frame

the required inputs are :

Real_heart.mat

Measured.mat

transfermatrix.mat
==========================


"""

from __future__ import print_function, division

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io
import cv2
import math
import pandas as pd
import statsmodels.formula.api as smf
import array
import scipy
from scipy import linalg


def generate(t_index, data):
    '''
    Generates Z data for the points in the X, Y meshgrid and parameter phi.
    '''
    signals,tsamples = data.shape
    print(math.sqrt(signals))
    W = int(math.sqrt(signals))

    print(type(t_index))
    print(t_index)

    
    Z = data[:,t_index]
    Z = Z.reshape((W,W))

    return Z


def error(real, estimate):

    square = (real- estimate)**2

    ava = square.mean()

    return ava 



def gen_invert(A, b, k, l):

    u, s, v = linalg.svd(A, full_matrices=False) #compute SVD without 0 singular values


    T,W = A.shape
    W = int(math.sqrt(W))
    #number of `columns` in the solution s, or length of diagnol
	
    S = np.diag(s)
    sr, sc = S.shape          #dimension of


    for i in range(0,sc-1):
	    if S[i,i]>0.00001:
    		S[i,i]=(1/S[i,i]) - (1/S[i,i])*(l/(l+S[i,i]**2))**k

    x1=np.dot(v.transpose(),S)    #why traspose? because svd returns v.transpose() but we need v
    x2=np.dot(x1,u.transpose())
    x3=np.dot(x2,b)

    #make the frame appliable in our coordinate
    x3 = (x3 / np.amax(np.absolute(x3)))
    #print(np.amax(heart))
    #print(np.amin(heart))
    heart_min = np.amin(x3)
    heart_max = np.amax(x3)

    x3 = x3.reshape((W,W))

    return(x3) 



# load my data
dic = scipy.io.loadmat('Real_heart.mat')
heart = dic['V_frame']

dicc = scipy.io.loadmat('Measured.mat')
signal = dicc['CardiacSignal']

trans = scipy.io.loadmat('transfermatrix.mat')
A = trans['transfer_matrix']

print(type(heart))
print(heart.shape)


# these are the parameters that will be used to for tikhonov
l=1
k=20


heart_V,tsamples = heart.shape

w = math.sqrt(heart_V)
print(w)

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
plt.title('Actual heart simulate Voltage')
plt.ylabel('heart cells labels/positions')

bx = fig.add_subplot(222, projection='3d')
plt.title('estimated voltage by tikhonov regularization')
plt.ylabel('heart cells labels/positions')

cx = fig.add_subplot(313)
plt.title('the edge heart cell action potential versus time')
plt.ylabel('Normalized Action potential ')
plt.xlabel('time in samples')


#print(np.amax(np.absolute(heart)))
#
heart = heart / np.amax(np.absolute(heart))
print(np.amax(heart))
print(np.amin(heart))
heart_min = np.amin(heart)
heart_max = np.amax(heart)


# process of cardiac signals got
#signal = signal / np.amax(np.absolute(signal))
#print(np.amax(signal))
#print(np.amin(signal))
#heart_min = np.amin(signal)
#heart_max = np.amax(signal)




# Plot cardiac signal
EGM = heart[1,:]

alltime    = len(EGM)
array_time = np.linspace(0,alltime,len(EGM))

cx.plot(array_time,EGM)




# Make the X, Y meshgrid.
xs = np.linspace(0, w-1, w)
ys = np.linspace(0, w-1, w)
W = int(w)
X, Y = np.meshgrid(xs, ys)



# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-1, 1)
bx.set_zlim(-1, 1)



# Begin plotting.
wframe = None
pframe = None
tframe = None

tstart = time.time()

for ts in np.linspace(0, tsamples-1, tsamples):
    # If a line collection is already remove it before drawing.
    if wframe:
        ax.collections.remove(wframe)

    if pframe:
        bx.collections.remove(pframe)

    #if tframe:
        #cx.collections.remove(tframe)
    
    #time instance
    print(type(ts))
    print(ts)
    tim = int(ts)
    print(type(tim))


    # all the data that we already known
    b = signal[:,tim]
    # Xh = heart[:,tim]


    # the real and estimated data
    Z = generate( tim, heart)
    Zp = gen_invert(A , b , k , l)

    print(type(Z))
    print(Z.shape)
    print(type(Zp))
    print(Zp.shape)
    #Plot the new wireframe and pause briefly before continuing.


    #calculate error character   
    err = error(Z,Zp)

    #reveal the errors
    fig.suptitle(u"Current time of the frame is {}ms, avarage square error is {}".format(tim, err),fontsize=16)

    
    #wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    #pframe = bx.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    pframe = bx.plot_surface(X, Y, Zp, rstride=1, cstride=1, cmap = 'cool')
    wframe = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = 'hot')
    #tframe = cx.settitle('the edge heart cell action potential versus time')
    
    plt.pause(.001)

print('Average FPS: %f' % (100 / (time.time() - tstart)))




