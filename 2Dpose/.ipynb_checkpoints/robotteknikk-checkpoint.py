#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:07:07 2018

@author: alexalcocer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib as mpl



def trplot2(T,name='',c='b'):  
    t = T[0:2,2] # translation
    R = T[:2,:2] # rotation  
    X = t + R[:,0] # 
    Y = t + R[:,1]  
    dtext = 0.1
    plt.plot([t[0],X[0]],[t[1],X[1]],color=c,linewidth=2.0)
    plt.plot([t[0],Y[0]],[t[1],Y[1]],color=c,linewidth=2.0)
    plt.text(t[0]-dtext,t[1]-dtext,"{"+name+"}",fontsize=14)
    plt.text(Y[0]+dtext/2,Y[1],r'$Y_{}$'.format(name),fontsize=14)
    plt.text(X[0]+dtext/2,X[1],r'$X_{}$'.format(name),fontsize=14)
    return None
 
def se2(x,y,theta):  
    #theta = np.random.rand()
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    t = np.array([[x],[y]])
    #q = np.concatenate([R,t], axis=1)
    #T = np.concatenate([q,[[0,0,1]]])
    T = np.block([[R,t],[0,0,1]])
    return T

def skew(w):
    """returns a skew symmetric matric from a vector
    w is 3x1 array"""
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    return np.array([[0,-w3,w2],[w3,0,-w1],[-w2,w1,0]])   


def trplot3(ax,T,name=None):
    """    T is 3D homogeneous transformation matrix T = [R p; 0 1]
    plots 3 vectors reference frame centered in p and orientation R
    p is a 3 array - origin of frame
    R is a 3x3 array - rotation matrix
    assumes that a figure is currently open with axes ax
    """
    R = T[0:3,0:3] # roation matrix
    p = T[0:3,3]
    X = R + p[:,np.newaxis] # X contains beacon coordinates in "inertial" frame
    dtext = 0.1
    ax.scatter(X[0,:],X[1,:],X[2,:],zdir='z', s=20) # plots beacon positions
    plt.plot([p[0], X[0,0]],[p[1],  X[1,0]],[p[2], X[2,0]],'r',linewidth=2)
    plt.plot([p[0], X[0,1]],[p[1],  X[1,1]],[p[2], X[2,1]],'g',linewidth=2)
    plt.plot([p[0], X[0,2]],[p[1],  X[1,2]],[p[2], X[2,2]],'b',linewidth=2)
    #ax.text(X[0]+dtext/2,X[1],r'$X_{}$'.format(name),fontsize=14)
    if name is not None:
        ax.text(p[0]-dtext,p[1]-dtext,p[2]-dtext, "{"+name+"}",fontsize=12)
        ax.text(X[0,0]+dtext/2,X[1,0],X[2,0], "$X_{}$".format(name),fontsize=10)
        ax.text(X[0,1]+dtext/2,X[1,1],X[2,1], "$Y_{}$".format(name),fontsize=10)
        ax.text(X[0,2]+dtext/2,X[1,2],X[2,2], "$Z_{}$".format(name),fontsize=10)

def e2h(p):
    "Euclidean to homogeneous coordinates"
    # adds a 1, works for 3D and 2D arrays
    return np.append(p,1)

def h2e(ph):
    "Homogeneous to Euclidean coordinates"
    # removes last coordinate, works for 3D and 2D arrays
    return ph[:-1]


def rotx(theta):
    # 3D rotation matrix along x axis
    return np.array([[1,0,0],
                     [0,np.cos(theta),-np.sin(theta)],
                     [0,np.sin(theta),np.cos(theta)]])

def roty(theta):
    # 3D rotation matrix along y axis
    return np.array([[np.cos(theta),0,np.sin(theta)],
                     [0,1,0],
                     [-np.sin(theta),0, np.cos(theta)]])    
    
def rotz(theta):
    # 3D rotation matrix along z axis
    return np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta),np.cos(theta),0],
                     [0,0,1]])


def trotx(theta):
    # 3D homogeneous transform matrix rotation matrix along x axis
    R = rotx(theta)
    T = np.eye(4)
    T[0:3,0:3] = R
    return T

def troty(theta):
    # 3D homogeneous transform matrix rotation matrix along x axis
    R = roty(theta)
    T = np.eye(4)
    T[0:3,0:3] = R
    return T

def trotz(theta):
    # 3D homogeneous transform matrix rotation matrix along x axis
    R = rotz(theta)
    T = np.eye(4)
    T[0:3,0:3] = R
    return T

def ttrans(p):
    # 3D homogeneous transform matrix pure translation
    T = np.eye(4)
    T[0:3,3] = p
    return T


def plot_ship(x, theta):
    # Vertex coordinates
    L1 = 3
    L2 = 6
    L3 = 7
    L = L1+L2+L3
    N = 0.2
    W1 = 2.3
    W2 = 2.5
    T = 2.3
    vert = np.array([[0, 0],[-N, 0], [-W1 ,-L1], [-W2 ,-L1-L2],[-T,-L1-L2-L3]])
    vert = vert + np.array([0,L])
    n = vert.shape[0]
    patch = np.zeros((2*n,2))
    patch[:n,:] = vert
    for i in range(n):
        patch[n+i,:] = np.array([-vert[n-i-1,0],vert[n-i-1,1]])
    polygon = patches.Polygon(patch, color="0.5", alpha=1) 
    r = mpl.transforms.Affine2D().rotate(-theta)
    t = mpl.transforms.Affine2D().translate(x[0],x[1])
    tra = r + t + ax.transData
    polygon.set_transform(tra)
    ax.add_patch(polygon)
    
